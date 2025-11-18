# GitHub Release Creation Guide

Follow these steps to create a GitHub Release with the inscription images.

## Step 1: Commit and Push Current Changes

```bash
cd /Users/shaigordin/Dropbox/Git-projects/safaitic-ocr

# Check what needs to be committed
git status

# Add the updated .gitignore
git add .gitignore

# Commit the changes
git commit -m "chore: Update .gitignore to exclude image zip from repo

The inscription_images.zip file (612 MB) will be hosted on GitHub
Releases to keep the repository size small."

# Push to GitHub
git push origin main
```

## Step 2: Create GitHub Release via Web Interface

1. **Go to your repository on GitHub**:
   - Navigate to: https://github.com/shaigordin/safaitic-ocr

2. **Click on "Releases"** (right sidebar or under "Code" tab)

3. **Click "Draft a new release"**

4. **Fill in release details**:
   - **Tag version**: `v1.0.0`
   - **Release title**: `v1.0.0 - Safaitic VLM Analysis with Interactive Demo`
   - **Description**: Copy content from `RELEASE_NOTES.md` (just created)

5. **Upload the image zip**:
   - Click "Attach binaries by dropping them here or selecting them"
   - Select: `docs/inscription_images.zip` (612.7 MB)
   - Wait for upload to complete (may take a few minutes)

6. **Publish release**:
   - Check "Set as the latest release"
   - Click "Publish release"

## Step 3: Create Release via GitHub CLI (Alternative)

If you have GitHub CLI (`gh`) installed:

```bash
# Install gh if needed:
# brew install gh

# Authenticate (if not already)
gh auth login

# Create the release
gh release create v1.0.0 \
  --title "v1.0.0 - Safaitic VLM Analysis with Interactive Demo" \
  --notes-file RELEASE_NOTES.md \
  docs/inscription_images.zip

# The zip file will be uploaded automatically
```

## Step 4: Update Main README with Release Link

Add a section to your main README.md pointing users to the release:

```markdown
## 🖼️ Inscription Images

High-resolution images of the 50 evaluated inscriptions are available in the [v1.0.0 Release](https://github.com/shaigordin/safaitic-ocr/releases/tag/v1.0.0).

**Download**: `inscription_images.zip` (612.7 MB, 284 images)

### Setup for Web Demo

1. Download the zip from the release
2. Extract to `docs/images/` directory
3. Follow setup instructions in [docs/README.md](docs/README.md)
```

## Step 5: Verify Release

After creating the release:

1. **Visit release page**: 
   https://github.com/shaigordin/safaitic-ocr/releases/tag/v1.0.0

2. **Verify zip is attached** and downloadable

3. **Test the download link**:
   ```bash
   # Direct download URL will be:
   curl -L -o test_download.zip \
     https://github.com/shaigordin/safaitic-ocr/releases/download/v1.0.0/inscription_images.zip
   ```

4. **Update docs/README.md** with the actual download link

## Step 6: Update Documentation with Release Link

Once release is published, update docs/README.md:

```markdown
### Option 3: Download from GitHub Releases (Recommended)

1. Download the latest release:
   ```bash
   curl -L -o inscription_images.zip \
     https://github.com/shaigordin/safaitic-ocr/releases/download/v1.0.0/inscription_images.zip
   ```

2. Extract to docs/images:
   ```bash
   cd docs
   unzip inscription_images.zip -d images/
   ```

3. Update image paths in index.html (see instructions below)
```

## File Locations

- **Image zip**: `/Users/shaigordin/Dropbox/Git-projects/safaitic-ocr/docs/inscription_images.zip`
- **Release notes**: `/Users/shaigordin/Dropbox/Git-projects/safaitic-ocr/RELEASE_NOTES.md`
- **This guide**: `/Users/shaigordin/Dropbox/Git-projects/safaitic-ocr/GITHUB_RELEASE_GUIDE.md`

## Troubleshooting

### Upload is too slow
- Use `gh` CLI instead of web interface
- Or use `git-lfs` to push large files

### Zip is ignored by git
- This is intentional! The .gitignore excludes it.
- Only upload to GitHub Releases, not commit to repo.

### Can't create release
- Make sure you're authenticated: `gh auth status`
- Or use web interface at github.com

## Next Steps After Release

1. ✅ Create release with image zip
2. ✅ Update README.md with release link
3. ✅ Test download from release page
4. ✅ Update docs/README.md with direct download URL
5. ✅ Announce release (optional: in README, social media, etc.)

## Benefits of GitHub Releases

- ✅ **Small repository size**: Main repo stays under 100 MB
- ✅ **Fast cloning**: Users can clone without downloading images
- ✅ **Versioned assets**: Images tied to specific release
- ✅ **Easy downloads**: Direct download links for users
- ✅ **Free hosting**: GitHub provides unlimited bandwidth for releases
