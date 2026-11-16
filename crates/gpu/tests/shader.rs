#[test]
fn shader_validates_without_a_gpu() {
    let module = naga::front::wgsl::parse_str(include_str!("../src/shader.wgsl")).unwrap();
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(),naga::valid::Capabilities::all()).validate(&module).unwrap();
}
