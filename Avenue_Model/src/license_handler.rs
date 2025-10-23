use std::sync::OnceLock;
use chrono::{DateTime, Duration,Utc};
use serde::{Serialize, Deserialize};
use ring::signature;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
struct License {
    organization: String,
    expiration: DateTime<Utc>,
}

#[derive(Clone)]
pub struct LicenseState {
    license: License,
    last_validated: DateTime<Utc>,
    signature: Vec<u8>,
    raw_data: Vec<u8>,
}

static LICENSE_STATE: OnceLock<Mutex<Option<LicenseState>>> = OnceLock::new();

const PUBLIC_KEY: &[u8] = &[212, 130, 119, 210, 45, 5, 192, 66, 24, 58, 51, 91, 107, 172, 108, 83, 12, 16, 141, 63, 121, 29, 53, 240, 135, 144, 226, 230, 105, 111, 180, 203];

const REVALIDATION_PERIOD: Duration = Duration::hours(24);

pub fn verify_license_internal(license_key: &str) -> Option<LicenseState> {
    let parts: Vec<&str> = license_key.split("::").collect();
    if parts.len() != 2 {
        return None;
    }

    let license_data = BASE64.decode(parts[0]).ok()?;
    let signature = BASE64.decode(parts[1]).ok()?;

    let public_key = signature::UnparsedPublicKey::new(
        &signature::ED25519,
        PUBLIC_KEY
    );

    // Verify signature
    public_key.verify(&license_data, &signature).ok()?;
    
    // Parse and validate license
    let license: License = serde_json::from_slice(&license_data).ok()?;
    if license.expiration <= Utc::now() {
        return None;
    }

    Some(LicenseState {
        license,
        last_validated: Utc::now(),
        signature,
        raw_data: license_data,
    })
}

pub fn validate_current_license() -> bool {
    true
    /*
    let state = LICENSE_STATE.get_or_init(|| Mutex::new(None));
    let mut state_guard = state.lock().unwrap();
    
    if let Some(license_state) = state_guard.as_ref() {
        // Check if we need to revalidate
        if Utc::now() - license_state.last_validated > REVALIDATION_PERIOD {
            // Revalidate the signature and expiration
            let public_key = signature::UnparsedPublicKey::new(
                &signature::ED25519,
                PUBLIC_KEY
            );
            
            if public_key.verify(&license_state.raw_data, &license_state.signature).is_err() {
                *state_guard = None;
                return false;
            }
            
            if license_state.license.expiration <= Utc::now() {
                *state_guard = None;
                return false;
            }
        }
        true
    } else {
        false
    }
    */
}

pub fn internal_initialize_license(license_key: &str) -> bool {
    let state = LICENSE_STATE.get_or_init(|| Mutex::new(None));
    let mut state_guard = state.lock().unwrap();
    match verify_license_internal(license_key) {
        Some(license_state) => {
            *state_guard = Some(license_state);
            true
        },
        None => false
    }
}