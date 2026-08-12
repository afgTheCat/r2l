use std::{error::Error as StdError, fmt, path::PathBuf};

/// A thread-safe, type-erased error used for failures originating outside
/// `r2l-core`.
pub type BoxedError = Box<dyn StdError + Send + Sync + 'static>;

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

/// Data type of a parameter value.
#[derive(Debug)]
pub enum DType {
    /// A 32-bit floating-point value.
    F32,
    /// An unsigned pointer-sized integer value.
    Usize,
    /// Another data type.
    Other(String),
}

/// A formatted parameter value and its data type.
#[derive(Debug)]
pub struct ValueWithDtype {
    /// Data type of the value.
    pub r#type: DType,
    /// Formatted value.
    pub value: String,
}

/// Reason a parameter is invalid.
#[derive(Debug, thiserror::Error)]
pub enum InvalidParameterError {
    /// A value does not satisfy the parameter's requirements.
    #[error("invalid value for `{name}`: expected {expected}, got {value}")]
    InvalidValue {
        /// Parameter name.
        name: String,
        /// Description of accepted values.
        expected: String,
        /// Supplied value.
        value: String,
    },
    /// The value is outside the accepted inclusive range.
    #[error("value {current_value:?} is outside the range {min:?}..={max:?}")]
    InvalidRange {
        /// Smallest accepted value.
        min: ValueWithDtype,
        /// Largest accepted value.
        max: ValueWithDtype,
        /// Supplied value.
        current_value: ValueWithDtype,
    },
    /// The supplied path is not valid for the parameter.
    #[error("invalid path: {path}", path = .path.display())]
    InvalidPath {
        /// Supplied path.
        path: PathBuf,
    },
}

/// Reason an artifact cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum BrokenArtifact {
    /// A required artifact does not exist.
    #[error("missing {artifact_type} artifact at {path}", path = .path.display())]
    Missing {
        /// Expected artifact path.
        path: PathBuf,
        /// Kind of artifact that was expected.
        artifact_type: String,
    },
    /// An artifact could not be decoded.
    #[error("failed to decode {artifact_type} artifact at {path}: {source}", path = .path.display())]
    Decode {
        /// Artifact path.
        path: PathBuf,
        /// Kind of artifact being decoded.
        artifact_type: String,
        /// Underlying decoder error.
        #[source]
        source: BoxedError,
    },
}

/// One or more artifacts that cannot be used.
#[derive(Debug)]
pub struct BrokenArtifacts {
    /// Broken artifacts discovered during validation or loading.
    pub broken: Vec<BrokenArtifact>,
}

impl fmt::Display for BrokenArtifacts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, artifact) in self.broken.iter().enumerate() {
            if index > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{artifact}")?;
        }
        Ok(())
    }
}

impl StdError for BrokenArtifacts {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.broken
            .first()
            .map(|artifact| artifact as &(dyn StdError + 'static))
    }
}

impl From<BrokenArtifact> for Error {
    fn from(artifact: BrokenArtifact) -> Self {
        Self::BrokenArtifacts(BrokenArtifacts {
            broken: vec![artifact],
        })
    }
}

/// A dependency required for an operation is unavailable.
#[derive(Debug, thiserror::Error)]
#[error("missing {dependency_type} dependency `{name}`")]
pub struct MissingDependency {
    /// Dependency name.
    pub name: String,
    /// Kind of dependency, such as a library, feature, or executable.
    pub dependency_type: String,
}

/// An environment operation failed.
#[derive(Debug, thiserror::Error)]
#[error("environment operation `{operation}` failed: {source}")]
pub struct EnvironmentError {
    /// Operation that failed, such as building, resetting, or stepping.
    pub operation: String,
    /// Underlying environment error.
    #[source]
    pub source: BoxedError,
}

/// An external resource stopped before completing an operation.
#[derive(Debug, thiserror::Error)]
#[error("{resource} was interrupted: {details}")]
pub struct ResourceInterrupted {
    /// Resource that was interrupted.
    pub resource: String,
    /// Additional failure details.
    pub details: String,
}

/// Errors shared by the `r2l` workspace.
#[derive(thiserror::Error)]
pub enum Error {
    /// A supplied parameter is invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(#[source] Box<InvalidParameterError>),

    /// An operation cannot run in the current state.
    #[error("invalid state for `{operation}`: {details}")]
    InvalidState {
        /// Operation that was requested.
        operation: String,
        /// Why the current state does not permit the operation.
        details: String,
    },

    /// The requested operation or capability is not supported.
    #[error("unsupported operation `{operation}`: {details}")]
    Unsupported {
        /// Unsupported operation or capability.
        operation: String,
        /// Additional context about the limitation.
        details: String,
    },

    /// One or more required artifacts are missing or cannot be decoded.
    #[error("broken artifacts: {0}")]
    BrokenArtifacts(#[source] BrokenArtifacts),

    /// A required dependency is unavailable.
    #[error(transparent)]
    MissingDependency(#[from] MissingDependency),

    /// An environment operation failed.
    #[error(transparent)]
    Environment(#[from] EnvironmentError),

    /// An external resource was interrupted.
    #[error(transparent)]
    ResourceInterrupted(#[from] ResourceInterrupted),

    /// A lower-level failure without a dedicated semantic category.
    #[error(transparent)]
    Wrapped(#[from] BoxedError),
}

impl Error {
    /// Wraps a lower-level error without discarding its source chain.
    pub fn wrap(error: impl StdError + Send + Sync + 'static) -> Self {
        Self::Wrapped(Box::new(error))
    }
}
