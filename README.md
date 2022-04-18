# cuosqp_isye6679

cpu version
gcc cpu_version_final.c -o cpu
./cpu
gpu version
nvcc main.cu -lcusparse -lcublas -o gpu
./gpu