import sys

__all__ = [
    'pt',
    'np',
]


def __dir__():
    import sys
    import types
    ret = super(types.ModuleType, sys.modules[__name__]).__dir__()
    return list(dict.fromkeys([*__all__, *ret]))  # drop duplicates


def __getattr__(item):
    import importlib

    if item in __all__:
        return importlib.import_module(f'{__package__}.{item}')
    else:
        class VerboseAttributeError(AttributeError):
            def __str__(self):
                if len(self.args) == 2 and isinstance(self.args[0], str):
                    import difflib
                    item, attributes = self.args
                    # Suggestions are sorted by their similarity.
                    suggestions = difflib.get_close_matches(
                        item, attributes, cutoff=0, n=100
                    )
                    return (
                        f'module {__package__} has no attribute {item!r}.\n'
                        f'Close matches: {suggestions!r}.')
                else:
                    return super().__str__()
        raise VerboseAttributeError(item, __dir__())


if sys.version_info < (3, 7):
    def _lazy_import_submodules(__path__, __name__, __package__):
        import sys
        import importlib

        class _LazySubModule(sys.modules[__name__].__class__):
            # In py37 is the class not necessary and __dir__ and __getattr__
            # are enough.
            # See: https://snarky.ca/lazy-importing-in-python-3-7
            def __dir__(self):
                return __dir__()

            def __getattr__(self, item):
                return __getattr__(item)

        sys.modules[__name__].__class__ = _LazySubModule


    _lazy_import_submodules(
        __name__=__name__, __path__=__path__, __package__=__package__
    )

del sys
