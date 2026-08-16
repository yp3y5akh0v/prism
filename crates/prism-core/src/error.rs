use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Errors returned when public PRISM inputs or persisted structures are invalid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrismError {
    /// A caller supplied an invalid runtime value or incompatible shape.
    InvalidInput(String),
    /// Persisted or encoded data violates the format's structural invariants.
    InvalidFormat(String),
    /// A size, offset, or identifier cannot be represented safely.
    Overflow(String),
}

impl Display for PrismError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid input: {message}"),
            Self::InvalidFormat(message) => write!(formatter, "invalid format: {message}"),
            Self::Overflow(message) => write!(formatter, "overflow: {message}"),
        }
    }
}

impl Error for PrismError {}

/// Result type used by fallible PRISM APIs.
pub type PrismResult<T> = Result<T, PrismError>;
