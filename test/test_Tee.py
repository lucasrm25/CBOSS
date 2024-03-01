import unittest
import sys
from pathlib import Path
from io import StringIO 
from CBOSS.utils.Tee import Tee, Tee2

class Test_Tee(unittest.TestCase):

    def test_stdout_stderr(self):
        msg_stderr = '\n... Testing stderr'
        msg_stdout = '\n... Testing stdout'
        
        T = Tee( Path(__file__).parent / 'log', Path(__file__).stem )
        with T as t:
            print(msg_stdout, end='')
            print(msg_stderr, file=sys.stderr, end='')

        with open(T.file_stdout.name, 'r') as f:
            f_read = f.read()
            self.assertTrue( f_read == msg_stdout, msg=f'\nExpected:\n\t{msg_stdout}\nGot:\n\t{f_read}' )
        with open(T.file_stderr.name, 'r') as f:
            f_read = f.read()
            self.assertTrue( f_read == msg_stderr,  msg=f'\nExpected:\n\t{msg_stderr}\nGot:\n\t{f_read}' )
        
class Test_Tee2(unittest.TestCase):

    def test_stdout_AND_stderr(self):

        msg_stdout = '\n... Testing stdout'
        msg_stderr = '\n... Testing stderr'

        file_stdout = Path(__file__).parent / 'log' / (Path(__file__).stem + '.' + self.__class__.__name__ + '.stdout.log')
        file_stderr = Path(__file__).parent / 'log' / (Path(__file__).stem + '.' + self.__class__.__name__ + '.stderr.log')
        str_stream_stdout  = StringIO()
        str_stream_stderr  = StringIO()

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

        ''' Assert Tee2 has saved the correct logs into the given stream files '''
        with (  
            open(file_stdout, 'r') as fstdout, 
            open(file_stderr, 'r') as fstderr
        ):
            f_read = fstdout.read()
            self.assertTrue( f_read == msg_stdout, msg=f'\nExpected:\n"{msg_stdout}"\nGot:\n"{f_read}"' )
            f_read = fstderr.read()
            self.assertTrue( f_read == msg_stderr,  msg=f'\nExpected:\n"{msg_stderr}"\nGot:\n"{f_read}"' )

        ''' Assert Tee2 has saved the correct logs into the given stream strings '''
        f_read = str_stream_stdout.getvalue()
        self.assertTrue( f_read == msg_stdout, msg=f'\nExpected:\n"{msg_stdout}"\nGot:\n"{f_read}"' )
        f_read = str_stream_stderr.getvalue()
        self.assertTrue( f_read == msg_stderr,  msg=f'\nExpected:\n"{msg_stderr}"\nGot:\n"{f_read}"' )

    def test_ONLY_stdout_NO_stderr(self):

        msg_stdout = '\n... Testing stdout'
        msg_stderr = '\n... Testing stderr'

        file_stdout = Path(__file__).parent / 'log' / (Path(__file__).stem + '.' + self.__class__.__name__ + '.stdout.log')
        file_stderr = Path(__file__).parent / 'log' / (Path(__file__).stem + '.' + self.__class__.__name__ + '.stderr.log')
        str_stream_stdout  = StringIO()
        str_stream_stderr  = StringIO()

        with (  
            open(file_stdout, 'w') as fstdout, 
            open(file_stderr, 'w') as fstderr,
            Tee2(
                stdout_streams=[fstdout, str_stream_stdout, sys.stdout], 
                # stderr_streams=[fstderr, str_stream_stderr, sys.stderr]
            ) as t
        ):
            print(msg_stdout, end='')
            print(msg_stderr, file=sys.stderr, end='')

        ''' Assert Tee2 has saved the correct logs into the given stream files '''
        with (  
            open(file_stdout, 'r') as fstdout, 
            open(file_stderr, 'r') as fstderr
        ):
            f_read = fstdout.read()
            self.assertTrue( f_read == msg_stdout + msg_stderr, msg=f'\nExpected:\n"{msg_stdout}"\nGot:\n"{f_read}"' )
            f_read = fstderr.read()
            self.assertTrue( f_read == "",  msg=f'\nExpected:\n"{msg_stderr}"\nGot:\n"{f_read}"' )

        ''' Assert Tee2 has saved the correct logs into the given stream strings '''
        f_read = str_stream_stdout.getvalue()
        self.assertTrue( f_read == msg_stdout + msg_stderr, msg=f'\nExpected:\n"{msg_stdout}"\nGot:\n"{f_read}"' )
        f_read = str_stream_stderr.getvalue()
        self.assertTrue( f_read == "",  msg=f'\nExpected:\n"{msg_stderr}"\nGot:\n"{f_read}"' )


if __name__ == "__main__":    

    # Test_Tee2().test_stdout_AND_stderr()
    # Test_Tee2().test_ONLY_stdout_NO_stderr()

    t = unittest.main(verbosity=2, exit=False, catchbreak=True)