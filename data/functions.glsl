layout(std140, binding = 0) buffer DataSsboIn {
  vec4 data[];
};

float readBuffer(uint pos) {
  return data[pos / 4][pos % 4];
}

void writeBuffer(uint pos, float val) {
  data[pos / 4][pos % pos] = val;
}

void matVecMultiply(uint mOffset, uint mCols, uint mRows, uint vOffset, uint vSize, uint rOffset) {
  uint index = gl_GlobalInvocationID.x;
  uint mRowOffset = index * mCols;

  float sum = 0;
  for (uint i = 0; i < mCols; ++i) {
    sum += readBuffer(mOffset + mRowOffset + i) * readBuffer(vOffset + i);
  }

  writeBuffer(rOffset + index, sum);
}

void vecVecAdd(uint aOffset, uint bOffset, uint size, uint rOffset) {
  uint index = gl_GlobalInvocationID.x;
  writeBuffer(rOffset + index, readBuffer(aOffset + index) + readBuffer(bOffset + index));
}

void vecScalarMultiply(uint vOffset, uint vSize, float x, uint rOffset) {
  uint index = gl_GlobalInvocationID.x;
  writeBuffer(rOffset + index, readBuffer(vOffset + index) * x);
}
