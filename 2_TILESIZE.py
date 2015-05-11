import os

#fix this to optimal number!
THREADCOUNT = 64
BLOCKCOUNT = 16

f = open('cudart.cu', 'r')
ll = list(f.readlines())

def tsdo(ts):
    for i, l in enumerate(ll):
        if l.startswith('#define THREADCOUNT'):
            ll[i] = '#define THREADCOUNT %d\n' % THREADCOUNT
        elif l.startswith('#define BLOCKCOUNT'):
            ll[i] = '#define BLOCKCOUNT %d\n' % BLOCKCOUNT
        elif l.startswith('#define TILE_WIDTH'):
            ll[i] = '#define TILE_WIDTH %d\n' % ts
        elif l.startswith('#define TILE_HEIGHT'):
            ll[i] = '#define TILE_HEIGHT %d\n' % ts
    path = 'cudart_TW_%d_TH_%d' % (ts, ts)
    ff = open('%s.cu' % path, 'w')
    ff.write(''.join(ll))
    ff.close()
    os.system('nvcc -O3 -lm -o %s %s.cu 2>%s_compile_log.txt' % (path, path, path))
    os.system('./%s room.scn > %s.txt 2>&1' % (path, path))
    print 'Tile size %d x %d complete' % (ts, ts)

for ts in (5, 10, 20, 25, 100):
    tsdo(ts)
