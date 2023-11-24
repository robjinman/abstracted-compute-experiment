layout(std140, binding = 0) buffer DataSsboIn {
  vec4 buffer[];
};

float readBuffer(uint pos) {
  return buffer[pos / 4][pos % 4];
}

void writeBuffer(uint pos, float val) {
  buffer[pos / 4][pos % pos] = val;
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
