(module
 (type $0 (func (param i32 i32 i32 i32 i32)))
 (global $assembly/kernels/PAGE_SIZE i32 (i32.const 65536))
 (memory $0 0)
 (export "PAGE_SIZE" (global $assembly/kernels/PAGE_SIZE))
 (export "binary_simd" (func $assembly/kernels/binary_simd))
 (export "int8_simd" (func $assembly/kernels/int8_simd))
 (export "fp32_simd" (func $assembly/kernels/fp32_simd))
 (export "memory" (memory $0))
 (func $assembly/kernels/int8_simd (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32) (param $4 i32)
  (local $5 v128)
  (local $6 i32)
  (local $7 i32)
  (local $8 i32)
  (local $9 v128)
  loop $for-loop|0
   local.get $3
   local.get $7
   i32.gt_s
   if
    local.get $1
    local.get $4
    local.get $7
    i32.mul
    i32.add
    local.set $8
    v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
    local.set $5
    i32.const 0
    local.set $6
    loop $for-loop|1
     local.get $6
     i32.const 128
     i32.lt_s
     if
      local.get $5
      local.get $6
      local.get $8
      i32.add
      v128.load
      local.tee $5
      i16x8.extend_low_i8x16_s
      local.get $0
      local.get $6
      i32.add
      v128.load
      local.tee $9
      i16x8.extend_low_i8x16_s
      i32x4.dot_i16x8_s
      i32x4.add
      local.get $5
      i16x8.extend_high_i8x16_s
      local.get $9
      i16x8.extend_high_i8x16_s
      i32x4.dot_i16x8_s
      i32x4.add
      local.set $5
      local.get $6
      i32.const 16
      i32.add
      local.set $6
      br $for-loop|1
     end
    end
    local.get $2
    local.get $7
    i32.const 2
    i32.shl
    i32.add
    local.get $5
    i32x4.extract_lane 0
    local.get $5
    i32x4.extract_lane 1
    i32.add
    local.get $5
    i32x4.extract_lane 2
    i32.add
    local.get $5
    i32x4.extract_lane 3
    i32.add
    i32.store
    local.get $7
    i32.const 1
    i32.add
    local.set $7
    br $for-loop|0
   end
  end
 )
 (func $assembly/kernels/fp32_simd (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32) (param $4 i32)
  (local $5 v128)
  (local $6 i32)
  (local $7 i32)
  (local $8 i32)
  loop $for-loop|0
   local.get $3
   local.get $7
   i32.gt_s
   if
    local.get $1
    local.get $4
    local.get $7
    i32.mul
    i32.add
    local.set $8
    v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
    local.set $5
    i32.const 0
    local.set $6
    loop $for-loop|1
     local.get $6
     i32.const 256
     i32.lt_s
     if
      local.get $5
      local.get $6
      local.get $8
      i32.add
      v128.load
      local.get $0
      local.get $6
      i32.add
      v128.load
      f32x4.mul
      f32x4.add
      local.set $5
      local.get $6
      i32.const 16
      i32.add
      local.set $6
      br $for-loop|1
     end
    end
    local.get $2
    local.get $7
    i32.const 2
    i32.shl
    i32.add
    local.get $5
    f32x4.extract_lane 0
    local.get $5
    f32x4.extract_lane 1
    f32.add
    local.get $5
    f32x4.extract_lane 2
    f32.add
    local.get $5
    f32x4.extract_lane 3
    f32.add
    f32.store
    local.get $7
    i32.const 1
    i32.add
    local.set $7
    br $for-loop|0
   end
  end
 )
 (func $assembly/kernels/binary_simd (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32) (param $4 i32)
  (local $5 v128)
  (local $6 i32)
  (local $7 i32)
  (local $8 i32)
  loop $for-loop|0
   local.get $3
   local.get $7
   i32.gt_s
   if
    local.get $1
    local.get $4
    local.get $7
    i32.mul
    i32.add
    local.set $8
    v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
    local.set $5
    i32.const 0
    local.set $6
    loop $for-loop|1
     local.get $6
     i32.const 128
     i32.lt_s
     if
      local.get $5
      local.get $6
      local.get $8
      i32.add
      v128.load
      local.get $0
      local.get $6
      i32.add
      v128.load
      v128.xor
      i8x16.popcnt
      i16x8.extadd_pairwise_i8x16_u
      i16x8.add
      local.set $5
      local.get $6
      i32.const 16
      i32.add
      local.set $6
      br $for-loop|1
     end
    end
    local.get $2
    local.get $7
    i32.const 2
    i32.shl
    i32.add
    local.get $5
    i16x8.extract_lane_u 0
    i32.const 65535
    i32.and
    local.get $5
    i16x8.extract_lane_u 1
    i32.const 65535
    i32.and
    i32.add
    local.get $5
    i16x8.extract_lane_u 2
    i32.const 65535
    i32.and
    i32.add
    local.get $5
    i16x8.extract_lane_u 3
    i32.const 65535
    i32.and
    i32.add
    local.get $5
    i16x8.extract_lane_u 4
    i32.const 65535
    i32.and
    i32.add
    local.get $5
    i16x8.extract_lane_u 5
    i32.const 65535
    i32.and
    i32.add
    local.get $5
    i16x8.extract_lane_u 6
    i32.const 65535
    i32.and
    i32.add
    local.get $5
    i16x8.extract_lane_u 7
    i32.const 65535
    i32.and
    i32.add
    i32.store
    local.get $7
    i32.const 1
    i32.add
    local.set $7
    br $for-loop|0
   end
  end
 )
)
