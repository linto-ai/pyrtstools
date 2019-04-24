import struct

def gen_wav_header(buffer: bytes,
                   sample_rate: int = 16000,
                   channels: int = 1,
                   sample_depth: int = 2) -> bytes:
    """ Return a wave header for the given buffer using given parameters

    Keyword Arguments:
    ==================
    buffer (bytes) -- audio buffer

    sample_rate (int) -- audio sample rate (default 16000)

    channels (int) -- number of channels (default 1)

    sample_depth (int) -- sample size in byte (default 2)
    """
    datasize = len(buffer)
    h = b'RIFF'                                                                                     
    h += struct.pack('<L4s4sLHHLLHH4s', 
                        datasize + 36,
                        b'WAVE', b'fmt ',
                        16, 1, channels, sample_rate,
                        sample_rate * channels * sample_depth,
                        channels * sample_depth, sample_depth * 8,
                        b'data')
    h += (datasize).to_bytes(4,'little')
    return h

def write_wav(buffer: bytes,
              file_path: str,
              sample_rate: int = 16000,
              channels: int = 1,
              sample_depth: int = 2):
    """ Write a wave file using the given parameters

    Keyword Arguments:
    ==================
    buffer (bytes) -- audio buffer

    file_path (str) -- the file to be written

    sample_rate (int) -- audio sample rate (default 16000)

    channels (int) -- number of channels (default 1)

    sample_depth (int) -- sample size in byte (default 2)
    """
    with open(file_path, 'wb') as f:
        f.write(gen_wav_header(buffer, sample_rate, channels, sample_depth) + buffer)
