//! License notices compiled into every binary and the wasm module, so that they ship with them
//! (`xf --licenses`, `xf-serve --licenses` / `/LICENSES.txt`, `xf-prep --licenses`).

macro_rules! head {
    () => {
        concat!(
            "Shiloh: an engine for The X-Files Game (1998). Not affiliated with or endorsed by the game's rights holders.\n",
            "Licensed under the MIT License or the Apache License, Version 2.0, at your option\n",
            "(LICENSE-MIT, LICENSE-APACHE in the source repository).\n\n",
            include_str!("../LICENSE-THIRD-PARTY"),
        )
    };
}

/// The project's license line followed by `port/LICENSE-THIRD-PARTY`.
#[cfg(not(feature = "qdmc"))]
pub const LICENSES: &str = head!();

/// Feature `qdmc`: also the qdmc crate's notice and the LGPL text it is distributed under.
#[cfg(feature = "qdmc")]
pub const LICENSES: &str = concat!(
    head!(),
    "\n================================================================================\n",
    "qdmc (this build: feature `qdmc`) — LGPL-2.1-or-later\n",
    "================================================================================\n\n",
    "The QDMC (QDesign Music) decoder of this build is the qdmc crate (directory qdmc/ of\n",
    "the source repository), a Rust port of FFmpeg's libavcodec/qdmc.c,\n",
    "Copyright (c) 2017 Paul B Mahol. It is free software under the GNU Lesser General\n",
    "Public License, version 2.1 or (at your option) any later version; its complete\n",
    "source is in the source repository. With this crate, the binary as a whole is\n",
    "distributed under the terms of that license. Its text follows.\n\n",
    include_str!("../../qdmc/COPYING"),
);

#[cfg(test)]
mod tests {
    #[test]
    fn notices_carry_the_third_party_texts() {
        assert!(super::LICENSES.contains("Copyright (c) 2026 Zeffuro"));
        assert!(super::LICENSES.contains("Copyright (c) 2021 Advanced Micro Devices"));
        assert!(super::LICENSES.contains("Copyright (C) 1997-2023 Sam Lantinga"));
        #[cfg(feature = "qdmc")]
        assert!(super::LICENSES.contains("Copyright (c) 2017 Paul B Mahol") && super::LICENSES.contains("LESSER GENERAL PUBLIC"));
    }
}
