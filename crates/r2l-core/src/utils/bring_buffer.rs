// This will probably be a separate crate published

use std::{
    alloc::{Layout, alloc, dealloc},
    mem::MaybeUninit,
    ptr::NonNull,
    slice,
};

// represents a single allocation.
struct BringBufferInner<T> {
    capacity: usize,
    data: NonNull<MaybeUninit<T>>,
}

impl<T> BringBufferInner<T> {
    fn new(capacity: usize) -> Self {
        const {
            assert!(
                std::mem::size_of::<T>() > 0,
                "BringBuffer does not support ZSTs"
            )
        };
        let layout = Layout::array::<T>(capacity)
            .expect("Allocation layot overflown isize::MAX. Big problem");
        let data = unsafe {
            let ptr = alloc(layout);
            NonNull::new(ptr)
                .expect("Null pointer returned by the allocator")
                .cast()
        };
        Self { capacity, data }
    }

    unsafe fn write(&mut self, offset: usize, value: T) {
        unsafe {
            self.data.add(offset).as_mut().write(value);
        }
    }

    // Has to make sure that the offset is initialized
    unsafe fn read(&self, offset: usize) -> &T {
        // Safety: caller must uphold the guarantee that the offset has been initialized
        unsafe { self.data.add(offset).as_ref().assume_init_ref() }
    }

    unsafe fn read_mut(&self, offset: usize) -> &mut T {
        unsafe { self.data.add(offset).as_mut().assume_init_mut() }
    }

    unsafe fn remove(&mut self, offset: usize) -> T {
        // Safety: caller must uphold the guarantee that the offset has been initialized
        unsafe { self.data.add(offset).read().assume_init() }
    }

    // Has to make sure that the offset is initialized
    unsafe fn overwrite(&mut self, offset: usize, value: T) -> T {
        unsafe {
            let old_value = self.remove(offset);
            self.write(offset, value);
            old_value
        }
    }

    unsafe fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data.cast().as_ptr(), self.capacity) }
    }
}

impl<T> Drop for BringBufferInner<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.data.as_ptr().cast(),
                Layout::array::<T>(self.capacity).unwrap(),
            );
        }
    }
}

// A ringbuffer that can be sliced only if we wrapped around exactly capacity times.
// Otherwise it supports
pub struct BringBuffer<T> {
    buffer: BringBufferInner<T>,
    // current length of the ringbuffer
    len: usize,
    // where ringbuffer[0] starts
    start_idx: usize,
}

impl<T> BringBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        let buffer = BringBufferInner::new(capacity);
        Self {
            buffer,
            len: 0,
            start_idx: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn read_last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            let last_idx = (self.start_idx + self.len - 1) % self.buffer.capacity;
            unsafe { Some(self.buffer.read(last_idx)) }
        }
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            let last_idx = (self.start_idx + self.len - 1) % self.buffer.capacity;
            let val = unsafe { self.buffer.remove(last_idx) };
            self.len -= 1;
            Some(val)
        }
    }

    pub fn back_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            None
        } else {
            let last_idx = (self.start_idx + self.len - 1) % self.buffer.capacity;
            let back = unsafe { self.buffer.read_mut(last_idx) };
            Some(back)
        }
    }

    pub fn index(&self, index: usize) -> Option<&T> {
        if index < self.len {
            let idx = (self.start_idx + index) % self.buffer.capacity;
            let val = unsafe { self.buffer.read(idx) };
            Some(val)
        } else {
            None
        }
    }

    pub fn index_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            let idx = (self.start_idx + index) % self.buffer.capacity;
            let val = unsafe { self.buffer.read_mut(idx) };
            Some(val)
        } else {
            None
        }
    }

    pub fn enqueu(&mut self, value: T) -> Option<T> {
        if self.len == self.buffer.capacity {
            let val = unsafe { self.buffer.overwrite(self.start_idx, value) };
            self.start_idx = (self.start_idx + 1) % self.buffer.capacity;
            Some(val)
        } else {
            let next_idx = (self.start_idx + self.len) % self.buffer.capacity;
            unsafe {
                self.buffer.write(next_idx, value);
            }
            self.len += 1;
            None
        }
    }

    // only if buffer if full and in order
    pub fn try_slice(&self) -> Option<&[T]> {
        if self.len == self.buffer.capacity && self.start_idx == 0 {
            unsafe { Some(self.buffer.as_slice()) }
        } else {
            None
        }
    }

    pub fn iter(&self) -> BringBufferIter<'_, T> {
        BringBufferIter::new(self)
    }
}

pub struct BringBufferIter<'a, T> {
    current: usize,
    bb: &'a BringBuffer<T>,
}

impl<'a, T> BringBufferIter<'a, T> {
    fn new(bb: &'a BringBuffer<T>) -> Self {
        Self { current: 0, bb }
    }
}

impl<'a, T> Iterator for BringBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.bb.len {
            return None;
        }
        let current = self.current;
        self.current += 1;
        self.bb.index(current)
    }
}

impl<T> Drop for BringBuffer<T> {
    fn drop(&mut self) {
        while self.pop_back().is_some() {}
    }
}
