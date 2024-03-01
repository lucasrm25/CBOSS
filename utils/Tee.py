import os, sys
from pathlib import Path
import traceback
import pdb
from io import StringIO 

class Multiple_osstreams(object):
    def __init__(self, stream_list:list):
        self.stream_list = stream_list

    def write(self, message):
        for s in self.stream_list:
            s.write(message)
        self.flush()
    
    def flush(self):
        for s in self.stream_list:
            s.flush()

    def fileno (self):
        return False

class Tee(object):
    # https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    
    def __init__(self, folder:str, name:str, print_stdout:bool=True, print_stderr:bool=True):
        self.print_stdout = print_stdout
        self.print_stderr = print_stderr
        # store original std streams
        self.stdout_org = sys.stdout
        self.stderr_org = sys.stderr
        
        # file_stdout = Path(folder) / name + '.stdout.log'
        # file_stderr = Path(folder) / name + '.stderr.log'
        os.makedirs(folder, exist_ok=True)

        self.file_stdout = open( Path(folder) / (name + '.stdout.log'), "w")
        self.file_stderr = open( Path(folder) / (name + '.stderr.log'), "w")


    def __enter__(self):
        # overwrite stdout and stderr channels
        sys.stdout = Multiple_osstreams(
            stream_list = [self.file_stdout] + [sys.stdout] * self.print_stdout
        )
        sys.stderr = Multiple_osstreams(
            stream_list = [self.file_stderr] + [sys.stderr] * self.print_stderr
        )
        return self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout.flush()
        sys.stderr.flush()
        # recover standard channels
        sys.stdout = self.stdout_org
        sys.stderr = self.stderr_org
        
        if exc_type is not None:
            self.file_stderr.write(traceback.format_exc())
        
        # close files
        self.file_stdout.close()
        self.file_stderr.close()
        return self


class Tee2(object):
    '''
    How to use:

    ```python
    str_stream_stdout = StringIO()
    str_stream_stderr = StringIO()
    with (  
        open(file_stdout, 'w') as fstdout, 
        open(file_stderr, 'w') as fstderr,
        Tee2(
            stdout_streams=[fstdout, str_stream_stdout, sys.stdout], 
            stderr_streams=[fstderr, str_stream_stderr, sys.stderr]
        ) as t
    ):
        print(msg_stdout, end='')
        print(msg_stderr, file=sys.stderr, end='')
    ```

    Reference: 
        - https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    '''
    
    def __init__(self, stdout_streams:list, stderr_streams:list=[]):
        # store original std streams
        self.stdout_org = sys.stdout
        self.stderr_org = sys.stderr

        self.stdout_streams = stdout_streams
        self.stderr_streams = stderr_streams if len(stderr_streams)>0 else stdout_streams

        assert isinstance(self.stdout_streams,list) and isinstance(self.stderr_streams,list)

    def __enter__(self):
        # overwrite stdout and stderr channels
        sys.stdout = Multiple_osstreams( stream_list = self.stdout_streams )
        sys.stderr = Multiple_osstreams( stream_list = self.stderr_streams )
        return self

    def __exit__(self, exc_type, exc_value, tb):
            
        if exc_type is not None:
            # print(traceback.format_exc(), file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, tb)
            
        sys.stdout.flush()
        sys.stderr.flush()
        # recover standard channels
        sys.stdout = self.stdout_org
        sys.stderr = self.stderr_org
        
        if exc_type is not None:
            # if exc_type is KeyboardInterrupt:
            #     raise KeyboardInterrupt(exc_type, exc_value, tb)
            # # pdb.post_mortem(tb)
            # else:
            #     raise Exception(exc_type, exc_value, tb)
            return False
            
        return True
    
    def closeall(self):
        ''' close all streams 
        '''
        for s in self.stdout_streams + self.stderr_streams:
            if not s.closed:
                s.close()