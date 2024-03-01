import json
import numpy as np
from pathlib import Path

class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small lists on single lines."""

    def __init__(self, indent:int=4, max_element_single_line:int=200, *args, **kwargs):
        super().__init__(indent=indent, *args, **kwargs)
        self.indentation_level = 0
        self.max_element_single_line = max_element_single_line
        self.indent = indent

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""

        if isinstance(o, (list, tuple)):
            if self._is_single_line_list(o):
                return "[" + ", ".join(json.dumps(el) for el in o) + "]"
            else:
                self.indentation_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

        elif isinstance(o, dict):
            self.indentation_level += 1
            output = [self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
            self.indentation_level -= 1
            return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

        elif isinstance(o, np.ndarray):
            return self.encode(o.tolist())
        elif isinstance(o, np.generic):
            return self.encode(o.item())
        elif isinstance(o, Path):
            return str(o)
        else:
            return json.dumps(o)

    def _is_single_line_list(self, o):
        if isinstance(o, (list, tuple)):
            return (
                not any(isinstance(el, (list, tuple, dict)) for el in o)
                and len(o) <= self.max_element_single_line
                # and len(str(o)) - 2 <= 60
            )

    @property
    def indent_str(self) -> str:
        return " " * self.indentation_level * self.indent
    
    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

