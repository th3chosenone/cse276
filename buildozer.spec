[app]

# ──────────────────────────────────────────────────────────────────────────────
# App Metadata
# ──────────────────────────────────────────────────────────────────────────────
title          = LastMileDeliveryOptimiser
package.name   = lastmiledelivery
package.domain = org.example

# Entry point
source.dir  = .
source.include_exts = py,png,jpg,jpeg,bmp,kv,atlas

version = 1.0.0

# ──────────────────────────────────────────────────────────────────────────────
# Requirements
# ──────────────────────────────────────────────────────────────────────────────
# NOTE: 'opencv' in buildozer maps to the kivy-bundled opencv recipe.
# numpy is a dependency of opencv and kivy; it will be pulled automatically.
requirements = python3,kivy==2.3.0,opencv,numpy

# ──────────────────────────────────────────────────────────────────────────────
# Orientation & Window
# ──────────────────────────────────────────────────────────────────────────────
orientation = landscape

# ──────────────────────────────────────────────────────────────────────────────
# Android-specific
# ──────────────────────────────────────────────────────────────────────────────
[buildozer]

# Android SDK / NDK managed by buildozer
android.sdk_path   =
android.ndk_path   =
android.ndk_api    = 21
android.minapi     = 21
android.api        = 33

# Architecture — target modern Android devices
android.archs = arm64-v8a, armeabi-v7a

# Android permissions
android.permissions = READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, INTERNET

# Allow cleartext traffic (not needed for this offline app, kept for debug)
android.manifest.intent_filters =

# Release keystore (fill in before production release)
# android.keystore      = /path/to/my.keystore
# android.keystore_pass = mypassword
# android.keyalias      = myalias
# android.keyalias_pass = mypassword

# ──────────────────────────────────────────────────────────────────────────────
# Log level
# ──────────────────────────────────────────────────────────────────────────────
log_level = 2

