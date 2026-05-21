use std::vec::Drain;

pub struct ReusableVec<T> {
    vec: Vec<T>,
}

impl<T> ReusableVec<T> {
    pub fn to_dropping_slice(&mut self) -> ReusableVecSlice<'_, T> {
        ReusableVecSlice {
            data: &mut self.vec,
        }
    }

    pub fn to_drain_iter(&mut self) -> ReusableVecDrainIter<'_, T> {
        ReusableVecDrainIter {
            drain: self.vec.drain(..),
        }
    }
}

pub struct ReusableVecSlice<'a, T> {
    data: &'a mut Vec<T>,
}

impl<'a, T> Drop for ReusableVecSlice<'a, T> {
    fn drop(&mut self) {
        self.data.clear();
    }
}

impl<'a, T> AsRef<[T]> for ReusableVecSlice<'a, T> {
    fn as_ref(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<'a, T> AsMut<[T]> for ReusableVecSlice<'a, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }
}

pub struct ReusableVecDrainIter<'a, T> {
    drain: Drain<'a, T>,
}

impl<'a, T> Iterator for ReusableVecDrainIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }
}

impl<'a, T> Drop for ReusableVecDrainIter<'a, T> {
    fn drop(&mut self) {
        while self.drain.next().is_some() {}
    }
}
