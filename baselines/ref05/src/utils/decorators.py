import errno
import os
import signal
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type, TypeVar

from src.utils.pylogger import RankedLogger

T = TypeVar("T")

logger = RankedLogger(__name__)


class TimedOutException(Exception):
    pass


class RetriesFailedException(Exception):
    pass


class __RetriableTimeoutException(Exception):
    pass

def timeout(
    seconds=10,
    error_message=os.strerror(errno.ETIME),
    timeout_action_func=None,
    exception_thrown_on_timeout=TimedOutException,
    **timeout_action_func_params,
):


    def decorator(func):
        def _handler(signum, frame):
            logger.info(error_message)
            if timeout_action_func:
                timeout_action_func(**timeout_action_func_params)
            raise exception_thrown_on_timeout()

        def wrapper(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:

                signal.alarm(0)

                signal.signal(signal.SIGALRM, old)
            return result

        return wraps(func)(wrapper)

    return decorator


def retry(
    exception_to_check: Type = Exception,
    tries: int = 5,
    delay_s: int = 3,
    backoff: int = 2,
    max_delay_s: Optional[int] = None,
    fn_execution_timeout_s: Optional[int] = None,
    deadline_s: Optional[int] = None,
    should_throw_original_exception: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:


    def deco_retry(f) -> Callable[..., T]:
        @wraps(f)
        def f_retry(*args, **kwargs) -> T:
            mtries, mdelay = tries, delay_s

            def fn(*args, **kwargs) -> T:
                if fn_execution_timeout_s is not None:
                    timeout_individual_fn_call_decorator = timeout(
                        seconds=fn_execution_timeout_s,
                        exception_thrown_on_timeout=__RetriableTimeoutException,
                    )
                    return timeout_individual_fn_call_decorator(f)(*args, **kwargs)
                return f(*args, **kwargs)

            acceptable_exceptions: Tuple[Type[Exception], ...] = (
                exception_to_check
                if isinstance(exception_to_check, tuple)
                else (exception_to_check,)
            )
            acceptable_exceptions = acceptable_exceptions + (
                __RetriableTimeoutException,
            )

            ret_val: T
            while mtries >= 0:
                try:
                    ret_val = fn(*args, **kwargs)
                    break
                except TimedOutException:
                    raise
                except acceptable_exceptions as e:
                    if mtries == 0:

                        logger.exception(f"Failed for the last time: {e}")
                        if should_throw_original_exception:
                            raise
                        raise RetriesFailedException(
                            f"Retry failed, permanently failing {f.__module__}:{f.__name__}, see logs for {e}"
                        )
                    msg = f"{e}, Retrying {f.__module__}:{f.__name__} in {mdelay} seconds..."
                    logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay = (
                        min(mdelay * backoff, max_delay_s)
                        if max_delay_s
                        else mdelay * backoff
                    )

            return ret_val

        if deadline_s is not None:
            global_retry_timeout_decorator = timeout(seconds=deadline_s)
            return global_retry_timeout_decorator(f_retry)

        return f_retry

    return deco_retry
