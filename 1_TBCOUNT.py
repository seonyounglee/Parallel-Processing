import os

f = open('cudart.cu', 'r')
ll = list(f.readlines())

def tcbc(tc, bc):
    for i, l in enumerate(ll):
        if l.startswith('#define THREADCOUNT'):
            ll[i] = '#define THREADCOUNT %d\n' % tc
        elif l.startswith('#define BLOCKCOUNT'):
            ll[i] = '#define BLOCKCOUNT %d\n' % bc
    path = 'cudart_THREADCOUNT_%d_BLOCKCOUNT_%d' % (tc, bc)
    ff = open('%s.cu' % path, 'w')
    ff.write(''.join(ll))
    ff.close()
    os.system('nvcc -O3 -lm -o %s %s.cu 2>%s_compile_log.txt' % (path, path, path))
    os.system('./%s room.scn > %s.txt 2>&1' % (path, path))
    print '<<<%d, %d>>> complete' % (bc, tc)

for tc in (16, 32, 64, 128, 256, 512, 1024):
    tcbc(tc, 1024/tc)
