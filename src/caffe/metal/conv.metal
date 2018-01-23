#include <metal_stdlib>
using namespace metal;
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w


kernel
void conv_bias(const device Dtype *im_in  [[buffer(0)]], device Dtype *im_out [[buffer(1)]], const device Dtype *wg     [[buffer(2)]], const device Dtype *bias     [[buffer(3)]], const constant int* int__[[buffer(4)]], uint3 tid[[thread_position_in_threadgroup]],uint3 off[[threadgroup_position_in_grid]]) {

int v_nax = int__[0];
int v_g = int__[1];
int v_B_off = int__[2];
int v_C_off = int__[3];
int v_imsi_0 = int__[4];
int v_imso_0 = int__[5];
int v_imsi_1 = int__[6];
int v_imso_1 = int__[7];
int v_imsi = int__[8];
int v_imso = int__[9];
int v_k_0 = int__[10];
int v_k_1 = int__[11];
int v_p_0 = int__[12];
int v_p_1 = int__[13];
int v_s_0 = int__[14];
int v_s_1 = int__[15];
int v_d_0 = int__[16];
int v_d_1 = int__[17];
int v_fin = int__[18];
int v_fout = int__[19];
int MG = int__[20];
int M = int__[21];
int N = int__[22];
int KG = int__[23];
int K = int__[24];
int v_num_tiles = int__[25];

const int tidn = tid.x;
const int tidm = tid.y;
const int offN = 32* off.x;
const int offM = 32* off.y;
threadgroup Dtype Asub[32][8];
threadgroup Dtype Bsub[8][32];
int batch = off.z;
const device Dtype* Aptr = wg;
const device Dtype* Bptr = im_in + v_B_off * batch;
device Dtype* Cptr = im_out + v_C_off * batch;
const device Dtype* Dptr = bias;
{
Dtype4 Creg[4][1];
#pragma unroll
for (int wm=0; wm<4; ++wm) {
#pragma unroll
for (int wn=0; wn<4/4; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < 4; ++la) {
int tid = tidm * 8 + tidn;
int id = la * 8 * 8 + tid;
int row = id / 8;
int col = id % 8;
int tiledIndex = 8 * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < 4; ++lb) {
int tid = tidm * 8 + tidn;
int id = lb * 8 * 8 + tid;
int col = id % 32;
int row = id / 32;
int tiledIndex = 8 * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
Dtype4 Areg;
Dtype4 Breg[1];
#pragma unroll 1
for (int kt=0; kt<8; kt+=1) {
#pragma unroll 1
for (int ku=0; ku<1; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<4/4; ++wn) {
int col = tidn + wn*4*8;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 8];
VEC_4_2(Breg[wn]) = Bsub[k][col + 16];
VEC_4_3(Breg[wn]) = Bsub[k][col + 24];
}
#pragma unroll
for (int wm=0; wm<4/4; ++wm) {
int row = tidm + wm*4*8;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 8][k];
VEC_4_2(Areg) = Asub[row + 16][k];
VEC_4_3(Areg) = Asub[row + 24][k];
#pragma unroll
for (int wn=0; wn<4/4; ++wn) {
VEC_4_0(Creg[wm * 4 + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * 4 + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * 4 + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * 4 + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * 4 + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * 4 + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * 4 + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * 4 + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * 4 + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * 4 + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * 4 + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * 4 + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * 4 + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * 4 + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * 4 + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * 4 + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

threadgroup_barrier(mem_flags::mem_threadgroup);
}
}
#pragma unroll
for (int wm=0; wm<4; ++wm) {
int globalRow = offM + tidm + wm * 8;
Dtype biasval = Dptr[globalRow];
#pragma unroll
for (int wn1=0; wn1<4; wn1+=4) {
int wn = wn1;
int globalCol = offN + tidn + wn * 8;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Creg[wm][wn / 4])).x + biasval;
}
wn++;
globalCol = offN + tidn + wn * 8;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Creg[wm][wn / 4])).y + biasval;
}
wn++;
globalCol = offN + tidn + wn * 8;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Creg[wm][wn / 4])).z + biasval;
}
wn++;
globalCol = offN + tidn + wn * 8;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Creg[wm][wn / 4])).w + biasval;
}
}
}
}
}

kernel
void conv_no_bias(const device Dtype *im_in  [[buffer(0)]], device Dtype *im_out [[buffer(1)]], const device Dtype *wg     [[buffer(2)]], const constant int* int__[[buffer(4)]], uint3 tid[[thread_position_in_threadgroup]],uint3 off[[threadgroup_position_in_grid]]) {


int v_nax = int__[0];
int v_g = int__[1];
int v_B_off = int__[2];
int v_C_off = int__[3];
int v_imsi_0 = int__[4];
int v_imso_0 = int__[5];
int v_imsi_1 = int__[6];
int v_imso_1 = int__[7];
int v_imsi = int__[8];
int v_imso = int__[9];
int v_k_0 = int__[10];
int v_k_1 = int__[11];
int v_p_0 = int__[12];
int v_p_1 = int__[13];
int v_s_0 = int__[14];
int v_s_1 = int__[15];
int v_d_0 = int__[16];
int v_d_1 = int__[17];
int v_fin = int__[18];
int v_fout = int__[19];
int MG = int__[20];
int M = int__[21];
int N = int__[22];
int KG = int__[23];
int K = int__[24];
int v_num_tiles = int__[25];


const int tidn = tid.x;
const int tidm = tid.y;
const int offN = 32* off.x;
const int offM = 32* off.y;
threadgroup Dtype Asub[32][8];
threadgroup Dtype Bsub[8][32];
int batch = off.z;
const device Dtype* Aptr = wg;
const device Dtype* Bptr = im_in + v_B_off * batch;
device Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[4][1];
#pragma unroll
for (int wm=0; wm<4; ++wm) {
#pragma unroll
for (int wn=0; wn<4/4; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < 4; ++la) {
int tid = tidm * 8 + tidn;
int id = la * 8 * 8 + tid;
int row = id / 8;
int col = id % 8;
int tiledIndex = 8 * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < 4; ++lb) {
int tid = tidm * 8 + tidn;
int id = lb * 8 * 8 + tid;
int col = id % 32;
int row = id / 32;
int tiledIndex = 8 * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
Dtype4 Areg;
Dtype4 Breg[1];
#pragma unroll 1
for (int kt=0; kt<8; kt+=1) {
#pragma unroll 1
for (int ku=0; ku<1; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<4/4; ++wn) {
int col = tidn + wn*4*8;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 8];
VEC_4_2(Breg[wn]) = Bsub[k][col + 16];
VEC_4_3(Breg[wn]) = Bsub[k][col + 24];
}
#pragma unroll
for (int wm=0; wm<4/4; ++wm) {
int row = tidm + wm*4*8;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 8][k];
VEC_4_2(Areg) = Asub[row + 16][k];
VEC_4_3(Areg) = Asub[row + 24][k];
#pragma unroll
for (int wn=0; wn<4/4; ++wn) {
VEC_4_0(Creg[wm * 4 + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * 4 + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * 4 + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * 4 + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * 4 + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * 4 + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * 4 + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * 4 + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * 4 + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * 4 + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * 4 + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * 4 + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * 4 + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * 4 + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * 4 + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * 4 + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

threadgroup_barrier(mem_flags::mem_threadgroup);
}
}
#pragma unroll
for (int wm=0; wm<4; ++wm) {
int globalRow = offM + tidm + wm * 8;
#pragma unroll
for (int wn1=0; wn1<4; wn1+=4) {
int wn = wn1;
int globalCol = offN + tidn + wn * 8;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Creg[wm][wn / 4])).x;
}
wn++;
globalCol = offN + tidn + wn * 8;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Creg[wm][wn / 4])).y;
}
wn++;
globalCol = offN + tidn + wn * 8;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Creg[wm][wn / 4])).z;
}
wn++;
globalCol = offN + tidn + wn * 8;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Creg[wm][wn / 4])).w;
}
}
}
}
}


