import transport_pb2

CHUNK_SIZE = 1024 * 1024

def save_chunks_to_file(chunks, filename):
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk.buffer)

def get_file_chunks(filename):
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield transport_pb2.Chunk(buffer=piece)
            

if __name__ == '__main__':
    pass