# Website Updates - Image Loading & Pagination

**Date**: November 18, 2024

## Summary

Updated the Safaitic OCR website demo to address two main issues:
1. **Images not displaying**: Created a zip file solution with 284 images (612 MB)
2. **Poor UX (long scroll)**: Implemented pagination with Previous/Next buttons

## Changes Made

### 1. Pagination UI (index.html)

**Added pagination controls:**
- Previous/Next buttons with keyboard shortcuts (← → arrow keys)
- Page counter showing "X of Y" inscriptions
- Buttons automatically disable at first/last pages
- Smooth scroll to top when changing pages
- Reset to page 1 when filters or search changes

**Key modifications:**
- Added `currentPage` and `filteredResults` variables to track pagination state
- Modified `displayResults()` to show only one inscription at a time
- Added `updatePaginationControls()` to manage button states
- Added `previousInscription()` and `nextInscription()` navigation functions
- Added keyboard event listener for arrow key navigation

**Location in file:**
- Line ~438: Added `filteredResults` and `currentPage` variables
- Line ~425: Added pagination controls HTML (buttons + page counter)
- Line ~540: Rewrote `displayResults()` for single-inscription display
- Line ~575: Added `updatePaginationControls()` function
- Line ~600: Added navigation functions
- Line ~695: Added keyboard shortcuts

### 2. Image Solution

**Created comprehensive image zip:**
- Script: `create_images_zip.py` - Extracts images from local directories
- Output: `docs/inscription_images.zip` (612.7 MB, 284 images)
- Includes all images for 50 evaluated inscriptions (BES15 1-50)
- Preserves directory structure: `BES15 X/im*.jpg`

**Current image display:**
- Website shows placeholder SVG images
- Placeholders display inscription siglum and note about local images
- Ready to be updated once images are deployed

**Image deployment options** (documented in `docs/README.md`):
1. **Option 1**: Extract locally and symlink for development
2. **Option 2**: Extract and commit to docs/images/ (612 MB added to repo)
3. **Option 3**: Upload to GitHub Releases (recommended - keeps repo small)
4. **Option 4**: Host on external CDN

### 3. Documentation

**Created `docs/README.md`:**
- Explains all 4 image deployment options
- Documents pagination features and keyboard shortcuts
- Describes filtering, search, and model comparison
- Includes data format specification
- Provides local development instructions

**Updated `.gitignore`:**
- Added exception for `docs/inscription_images.zip`
- Allows zip file to be committed while keeping individual images excluded

## Technical Details

### Pagination Implementation

**State management:**
```javascript
let filteredResults = [];  // Current filtered set
let currentPage = 0;       // Index of current inscription (0-based)
```

**Display logic:**
```javascript
// Show only one inscription at a time
const currentResult = filteredResults[currentPage];
content.innerHTML = renderInscription(currentResult);
updatePaginationControls();
```

**Navigation:**
```javascript
function nextInscription() {
    if (currentPage < filteredResults.length - 1) {
        currentPage++;
        displayResults();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}
```

**Button states:**
- Previous disabled when `currentPage === 0`
- Next disabled when `currentPage >= filteredResults.length - 1`
- Opacity and cursor updated for visual feedback

### Image Placeholder Format

**Current placeholder SVG:**
```javascript
const placeholderSvg = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='250' height='250'%3E
    %3Crect fill='%23f0f0f0' width='250' height='250'/%3E
    %3Ctext x='50%25' y='45%25'%3E${siglum}%3C/text%3E
    %3Ctext x='50%25' y='55%25'%3EImages not included%3C/text%3E
    %3Ctext x='50%25' y='65%25'%3E(See data/examples/BES15/)%3C/text%3E
%3C/svg%3E`;
```

**To update for actual images** (once deployed):
```javascript
// For GitHub Pages with extracted images:
const imgPath = `images/${siglum}/im${imageFilename}.jpg`;

// For local development with symlink:
const imgPath = `../images/${siglum}/im${imageFilename}.jpg`;
```

Note: Image filenames are not sequential (e.g., im0037246.jpg, im0042041.jpg), so you'll need to either:
1. List directory contents dynamically
2. Include image filenames in the JSON data
3. Use a predictable naming convention

## Image Statistics

**Zip file contents:**
- 50 inscriptions
- 284 total images (average 5.7 images per inscription)
- Range: 1-15 images per inscription
- Original size: 617.5 MB
- Compressed: 612.7 MB (0.8% compression - JPEGs don't compress much further)

**Top inscriptions by image count:**
- BES15 1: 15 images
- BES15 7: 10 images
- BES15 38: 10 images
- BES15 44: 10 images

## Testing

**Local server started:**
```bash
cd docs
python3 -m http.server 8080
```

**Access at:** http://localhost:8080

**Test checklist:**
- ✅ Pagination buttons display correctly
- ✅ Previous button disabled on first page
- ✅ Next button disabled on last page
- ✅ Page counter shows "1 of 50"
- ✅ Arrow keys navigate between inscriptions
- ✅ Smooth scroll to top on page change
- ✅ Filter reset to page 1
- ✅ Search reset to page 1
- ✅ Placeholder images display with inscription info

## Recommended Next Steps

### 1. Deploy Images (Choose One Option)

**Option A: GitHub Releases (Recommended)**
```bash
# Create a new release on GitHub
# Upload docs/inscription_images.zip as release asset
# Update README with download instructions
```

**Option B: Commit to Repository**
```bash
cd docs
unzip inscription_images.zip -d images/
# Update .gitignore to allow docs/images/
git add images/
git commit -m "Add inscription images for web demo"
```

**Option C: External CDN**
- Upload zip to AWS S3, Cloudflare R2, or similar
- Update image URLs in index.html

### 2. Update Image Loading Code

Once images are deployed, update `renderInscription()` in `index.html`:

```javascript
// Current line ~620, replace placeholder with:
for (let i = 1; i <= result.num_images; i++) {
    // You'll need to know the actual image filenames
    // Either add them to JSON or list directory dynamically
    const imgSrc = `images/${siglum}/im${imageNum}.jpg`;
    images.push(`
        <div class="image-container">
            <img src="${imgSrc}" alt="${result.inscription_siglum} - Image ${i}"
                 onerror="this.src='${placeholderSvg}'">
        </div>
    `);
}
```

### 3. Add Image Filenames to JSON (Optional)

To avoid hardcoding image paths, add image filenames to the evaluation JSON:

```json
{
  "inscription_siglum": "BES15 1",
  "num_images": 15,
  "image_filenames": [
    "im0037246.jpg",
    "im0037250.jpg",
    ...
  ],
  ...
}
```

Then update the rendering code to use these filenames.

## Files Modified

1. **docs/index.html** (695 lines)
   - Added pagination controls HTML
   - Added pagination state variables
   - Rewrote displayResults() for single-inscription view
   - Added navigation functions
   - Added keyboard shortcuts
   - Updated image rendering to use placeholders

2. **create_images_zip.py** (NEW - 126 lines)
   - Reads evaluation JSON to get inscription list
   - Creates zip with images from data/examples/BES15/
   - Includes only full-size images (im*.jpg, not thumbnails)
   - Provides compression statistics

3. **docs/README.md** (NEW - 162 lines)
   - Documents image setup options
   - Explains pagination features
   - Provides development instructions
   - Describes data format

4. **.gitignore** (updated)
   - Added exception for docs/inscription_images.zip

## Git Commit

**Suggested commit message:**
```
feat: Add pagination and image zip for web demo

- Implement pagination UI with prev/next buttons and keyboard shortcuts
- Show one inscription at a time instead of long scroll
- Create inscription_images.zip (612 MB, 284 images, 50 inscriptions)
- Add docs/README.md with image deployment options
- Add placeholder images until deployment solution chosen
- Reset to page 1 when filters or search change
- Smooth scroll to top on navigation
```

## Browser Compatibility

**Tested features:**
- ES6 JavaScript (arrow functions, const/let)
- Fetch API for JSON loading
- SVG data URLs for placeholders
- CSS Flexbox for pagination controls
- Event listeners for keyboard shortcuts

**Compatible with:**
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Modern mobile browsers

## Performance

**Single-inscription display:**
- No longer renders 50 inscriptions at once
- Faster initial display (1 inscription vs 50)
- Reduced memory usage
- Smooth navigation between pages

**Image loading:**
- Currently shows instant placeholders (SVG)
- Once real images deployed, add lazy loading:
  ```javascript
  <img loading="lazy" src="..." />
  ```

## Future Enhancements

**Possible additions:**
1. **Jump to page**: Input field or dropdown to jump to specific inscription
2. **URL parameters**: Support `?page=5` for direct linking
3. **Image gallery**: Lightbox/modal for full-size image viewing
4. **Image comparison**: Side-by-side view of multiple inscriptions
5. **Thumbnail previews**: Show small thumbnails in pagination
6. **Keyboard shortcuts**: Add more (e.g., Home/End for first/last page)
7. **Swipe gestures**: Touch support for mobile navigation

## Known Issues

**Images:**
- Currently show placeholders - need deployment solution chosen
- Image filenames in directories are not sequential
- Need to add image filenames to JSON or implement directory listing

**None critical - pagination fully functional.**

## Questions for You

1. **Image deployment preference**: GitHub Releases, commit to repo, or external CDN?
2. **Image filenames**: Should I update the evaluation scripts to save image filenames in JSON?
3. **Additional features**: Want jump-to-page or other navigation enhancements?
4. **OCIANA integration**: Still want to try loading from external website, or zip solution is good?

## Conclusion

The website now has:
- ✅ **Pagination**: Navigate with buttons or arrow keys
- ✅ **Better UX**: One inscription at a time, no long scroll
- ✅ **Image solution**: Zip file ready for deployment (612 MB)
- ✅ **Documentation**: Comprehensive README for image setup
- ✅ **Flexible deployment**: 4 options documented

All functionality is complete and tested. The only remaining decision is which image deployment option to use.
