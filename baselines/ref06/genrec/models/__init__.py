

__all__ = []


try:
    from genrec.models.DIFF_GRM.model import DIFF_GRM
except ImportError:
    DIFF_GRM = None
    print("Warning: DIFF_GRM model not found, skipping import")

try:
    from genrec.models.AR_GRM.model import AR_GRM
except ImportError:
    AR_GRM = None
    print("Warning: AR_GRM model not found, skipping import")


if DIFF_GRM is not None:
    __all__.append('DIFF_GRM')
if AR_GRM is not None:
    __all__.append('AR_GRM')
