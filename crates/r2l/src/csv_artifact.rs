use std::{
    fs::{File, OpenOptions},
    io::ErrorKind::NotFound,
    marker::PhantomData,
    path::Path,
};

use csv::{ReaderBuilder, WriterBuilder};
use r2l_core::error::{Error, Result};
use serde::{Serialize, de::DeserializeOwned};

/// CSV output that preserves existing data only after validating its schema and rows.
pub(crate) struct CsvArtifact<R> {
    writer: csv::Writer<File>,
    record: PhantomData<R>,
}

impl<R: Serialize + DeserializeOwned> CsvArtifact<R> {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().read(true).append(true).open(path);
        let writer = match file {
            Ok(mut file) if file.metadata()?.len() != 0 => {
                let mut reader = ReaderBuilder::new().from_reader(&mut file);
                for row in reader.deserialize::<R>() {
                    row.map_err(Error::wrap)?;
                }
                WriterBuilder::new().has_headers(false).from_writer(file)
            }
            Ok(file) => WriterBuilder::new().has_headers(true).from_writer(file),
            Err(error) if error.kind() == NotFound => csv::WriterBuilder::new()
                .has_headers(true)
                .from_path(path)
                .map_err(Error::wrap)?,
            Err(error) => return Err(error.into()),
        };
        Ok(Self {
            writer,
            record: PhantomData,
        })
    }

    pub(crate) fn write(&mut self, row: &R) -> Result<()> {
        self.writer.serialize(row).map_err(Error::wrap)?;
        self.writer.flush()?;
        Ok(())
    }
}
