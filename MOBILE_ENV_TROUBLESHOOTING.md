# Mobile App Environment Variables Troubleshooting Guide

## Common Issues with Supabase Not Connecting in APK Builds

### Issue: Environment Variables Not Loaded During Build

When building an APK, environment variables must be available at **build time**, not just runtime. Here are the most common causes and solutions:

---

## For React Native (Expo) Apps

### 1. **Check Environment Variable Prefix**

Expo requires the `EXPO_PUBLIC_` prefix for environment variables to be accessible in the client:

```env
# ❌ WRONG - Won't work in Expo
SUPABASE_URL=your-url
SUPABASE_ANON_KEY=your-key

# ✅ CORRECT - Works in Expo
EXPO_PUBLIC_SUPABASE_URL=your-url
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-key
```

### 2. **Configure app.config.js or app.json**

Your `app.config.js` must read environment variables:

```javascript
// app.config.js
require('dotenv').config();

export default {
  expo: {
    name: "Your App",
    // ... other config
    extra: {
      supabaseUrl: process.env.EXPO_PUBLIC_SUPABASE_URL,
      supabaseAnonKey: process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,
    },
  },
};
```

### 3. **Location of .env Files**

For Expo, place `.env` files in the **root of your Expo project** (same level as `app.json` or `app.config.js`):

```
your-project/
├── .env
├── .env.local
├── app.config.js
├── package.json
└── ...
```

### 4. **Install dotenv**

Make sure `dotenv` is installed:

```bash
npm install dotenv
```

### 5. **Build Command**

When building, ensure environment variables are loaded:

```bash
# For EAS Build
eas build --platform android

# For local build
npx expo prebuild
npx expo run:android
```

---

## For React Native (Bare/Custom) Apps

### 1. **Use react-native-config**

Install and configure `react-native-config`:

```bash
npm install react-native-config
cd ios && pod install && cd .. # For iOS
```

### 2. **Create .env Files**

Create `.env` files in the **root of your React Native project**:

```env
# .env
SUPABASE_URL=your-url
SUPABASE_ANON_KEY=your-key
```

### 3. **Access in Code**

```javascript
import Config from 'react-native-config';

const supabaseUrl = Config.SUPABASE_URL;
const supabaseKey = Config.SUPABASE_ANON_KEY;
```

### 4. **Android Configuration**

For Android, you may need to add to `android/app/build.gradle`:

```gradle
apply from: project(':react-native-config').projectDir.getPath() + "/dotenv.gradle"
```

---

## For Flutter Apps

### 1. **Use flutter_dotenv**

Add to `pubspec.yaml`:

```yaml
dependencies:
  flutter_dotenv: ^5.0.2
```

### 2. **Create .env File**

Create `.env` in the **root of your Flutter project**:

```env
SUPABASE_URL=your-url
SUPABASE_ANON_KEY=your-key
```

### 3. **Load in Code**

```dart
import 'package:flutter_dotenv/flutter_dotenv.dart';

await dotenv.load(fileName: ".env");
String supabaseUrl = dotenv.env['SUPABASE_URL']!;
```

### 4. **Include in Build**

Add `.env` to `pubspec.yaml` assets:

```yaml
flutter:
  assets:
    - .env
```

---

## General Troubleshooting Steps

### Step 1: Verify .env File Location

Your `.env` files should be in the **root directory of your mobile app project**, not in backend/frontend folders.

### Step 2: Check File Names

Common `.env` file names:
- `.env` (default)
- `.env.production` (for production builds)
- `.env.local` (local overrides)

### Step 3: Verify Environment Variables Are Set

Before building, check that variables are loaded:

**For Expo:**
```bash
# In your app code, add temporary logging:
console.log('Supabase URL:', process.env.EXPO_PUBLIC_SUPABASE_URL);
```

**For React Native:**
```bash
# Check if Config is working:
import Config from 'react-native-config';
console.log('Supabase URL:', Config.SUPABASE_URL);
```

### Step 4: Clean Build

Sometimes cached builds cause issues:

```bash
# For Expo
npx expo start -c

# For React Native
cd android && ./gradlew clean && cd ..
cd ios && pod deintegrate && pod install && cd ..

# For Flutter
flutter clean
flutter pub get
```

### Step 5: Check Build Logs

Look for environment variable loading messages in build logs. If you see `undefined` or empty values, the env vars aren't being loaded.

---

## Common Mistakes

1. **❌ Wrong Prefix**: Using `SUPABASE_URL` instead of `EXPO_PUBLIC_SUPABASE_URL` in Expo
2. **❌ Wrong Location**: `.env` file in wrong directory
3. **❌ Not Loading**: Forgetting to call `dotenv.config()` or similar
4. **❌ Gitignored**: `.env` file is gitignored (correct) but not copied to new laptop
5. **❌ Build Cache**: Old build cache with missing env vars
6. **❌ Case Sensitivity**: Environment variable names are case-sensitive

---

## Quick Checklist

- [ ] `.env` file exists in mobile app root directory
- [ ] Environment variables have correct prefix (`EXPO_PUBLIC_` for Expo)
- [ ] `app.config.js` or equivalent reads from `.env`
- [ ] `dotenv` or equivalent package is installed
- [ ] Clean build performed after adding env vars
- [ ] Variables are accessible in code (test with console.log)
- [ ] Build command includes environment loading

---

## Solution for Your Specific Case

Since you mentioned you have **4 env files** set up, make sure:

1. **All 4 `.env` files are in the correct location** (mobile app root, not backend/frontend)
2. **Environment variables use correct prefixes** for your framework
3. **Build configuration reads from `.env` files** (check `app.config.js`, `build.gradle`, etc.)
4. **Perform a clean build** after setting up env files
5. **Verify variables are loaded** by logging them in your app code

If you can share:
- Your mobile app framework (Expo, React Native, Flutter)
- The location of your `.env` files
- Your `app.config.js` or build configuration

I can provide more specific guidance!
