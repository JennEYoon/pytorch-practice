from documentation  
https://pytorch.org/docs/stable/generated/torch.load.html  

# torch.load

## defined  
torch.load(f, map_location=None, pickle_module=pickle, *, weights_only=False, mmap=None, **pickle_load_args)[source]
Loads an object saved with torch.save() from a file.

torch.load() uses Python’s unpickling facilities but treats storages, which underlie tensors, specially. They are first deserialized on the CPU and are then moved to the device they were saved from. If this fails (e.g. because the run time system doesn’t have certain devices), an exception is raised. However, storages can be dynamically remapped to an alternative set of devices using the map_location argument.

### Parameters
f (Union[str, PathLike, BinaryIO, IO[bytes]]) – a file-like object (has to implement read(), readline(), tell(), and seek()), or a string or os.PathLike object containing a file name

map_location (Optional[Union[Callable[[Storage, str], Storage], device, str, Dict[str, str]]]) – a function, torch.device, string or a dict specifying how to remap storage locations

pickle_module (Optional[Any]) – module used for unpickling metadata and objects (has to match the pickle_module used to serialize file)

weights_only (Optional[bool]) – Indicates whether unpickler should be restricted to loading only tensors, primitive types, dictionaries and any types added via torch.serialization.add_safe_globals().

mmap (Optional[bool]) – Indicates whether the file should be mmaped rather than loading all the storages into memory. Typically, tensor storages in the file will first be moved from disk to CPU memory, after which they are moved to the location that they were tagged with when saving, or specified by map_location. This second step is a no-op if the final location is CPU. When the mmap flag is set, instead of copying the tensor storages from disk to CPU memory in the first step, f is mmaped.

pickle_load_args (Any) – (Python 3 only) optional keyword arguments passed over to pickle_module.load() and pickle_module.Unpickler(), e.g., errors=....

### Return type
Any
