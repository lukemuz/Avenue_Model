
use crate::license_handler::{internal_initialize_license, validate_current_license};


fn read_license_key() -> String {
    // Read the license key from the license.txt file
    let license_key = std::fs::read_to_string("license_key.txt").expect("Failed to read license.txt");
    license_key
}

pub fn initialize_test_license() {
    let license_key = read_license_key();
    internal_initialize_license(&license_key);
    let is_valid = validate_current_license();
    assert!(is_valid);
}

#[test]
fn test_license() {
    println!("Current directory: {:?}", std::env::current_dir().unwrap());
    initialize_test_license();
}