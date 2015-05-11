import os

#fix this to optimal number!
THREADCOUNT = 64
BLOCKCOUNT = 16
TILESIZE = 25

f = open('cudart.cu', 'r')
ll = list(f.readlines())

def forloop(spk):
    for i, l in enumerate(ll):
        if l.startswith('#define THREADCOUNT'):
            ll[i] = '#define THREADCOUNT %d\n' % THREADCOUNT
        elif l.startswith('#define BLOCKCOUNT'):
            ll[i] = '#define BLOCKCOUNT %d\n' % BLOCKCOUNT
        elif l.startswith('#define TILE_WIDTH'):
            ll[i] = '#define TILE_WIDTH %d\n' % TILESIZE
        elif l.startswith('#define TILE_HEIGHT'):
            ll[i] = '#define TILE_HEIGHT %d\n' % TILESIZE
        elif l.startswith('#define RAYTRACE_SAMPLES_PER_KERNEL_EXECUTION'):
            ll[i] = '#define RAYTRACE_SAMPLES_PER_KERNEL_EXECUTION (THREADCOUNT * BLOCKCOUNT * %d)\n' % spk
    path = 'cudart_FOR_%d' % spk
    ff = open('%s.cu' % path, 'w')
    ff.write(''.join(ll))
    ff.close()
    os.system('nvcc -O3 -lm -o %s %s.cu 2>%s_compile_log.txt' % (path, path, path))
    os.system('./%s room.scn > %s.txt 2>&1' % (path, path))
    print '%d samples per thread in kernel complete' % spk

for spk in range(1, 17):
    forloop(spk)
