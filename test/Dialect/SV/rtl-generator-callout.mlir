// RUN:  circt-opt -rtl-generator-callout='schema-name=Schema_Name generator-executable=echo generator-executable-arguments=file1.v,file2.v,file3.v,file4.v generator-executable-output-filename=litTestout' %s | FileCheck %s

module attributes {firrtl.mainModule = "top_mod"}  {
  rtl.generator.schema @SchemaVar, "Schema_Name", ["port1", "port2"]
  rtl.module.generated @sampleModuleName, @SchemaVar(%rd_clock_0: i1, %rd_en_0: i1, %rd_addr_0: i4,
  %rd_clock_1: i1, %rd_en_1: i1, %rd_addr_1: i4) -> (%rd_data_0: i16, %rd_data_1: i16) attributes {port1 = 10 : i64, port2 = 2 : i32}

  // CHECK:  rtl.module.extern @sampleModuleName(%rd_clock_0: i1, %rd_en_0: i1, %rd_addr_0: i4, %rd_clock_1: i1, %rd_en_1: i1, %rd_addr_1: i4) -> (%rd_data_0: i16, %rd_data_1: i16) attributes {filenames =  "file1.v,file2.v,file3.v,file4.v --moduleName sampleModuleName --port1 10 --port2 2"}

  rtl.generator.schema @SchemaVar1, "Schema_Name1", ["port1", "port2"]
  rtl.module.generated @sampleModuleName1, @SchemaVar1(%rd_clock_0: i1, %rd_en_0: i1, %rd_addr_0: i4, %rd_clock_1: i1, %rd_en_1: i1, %rd_addr_1: i4) -> (%rd_data_0: i16, %rd_data_1: i16) attributes {port1 = 10 : i64, port2 = 2 : i32}
  // CHECK: rtl.module.generated @sampleModuleName1, @SchemaVar1(%rd_clock_0: i1, %rd_en_0: i1, %rd_addr_0: i4, %rd_clock_1: i1, %rd_en_1: i1, %rd_addr_1: i4) -> (%rd_data_0: i16, %rd_data_1: i16) attributes {port1 = 10 : i64, port2 = 2 : i32}
}

