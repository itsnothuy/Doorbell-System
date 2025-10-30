# PWA Icons Generation Guide

## Required Icons

The following icon sizes are required for full PWA support:
- 72x72px
- 96x96px
- 128x128px
- 144x144px
- 152x152px
- 192x192px
- 384x384px
- 512x512px

## Generation Steps

1. Create a base icon (1024x1024px recommended) with:
   - Doorbell/security camera symbol
   - Blue gradient background (#2563eb to #764ba2)
   - Clean, modern design

2. Use an image editor or online tool to resize to all required sizes

3. Save as PNG format with transparency where appropriate

4. Place all icons in `/static/images/` directory

## Temporary Placeholder

For development, you can use a simple colored square or the favicon from:
```bash
# Create simple placeholder icons using ImageMagick (if available)
for size in 72 96 128 144 152 192 384 512; do
    convert -size ${size}x${size} xc:#2563eb \
            -gravity center -pointsize $(($size/3)) \
            -fill white -annotate +0+0 "🔔" \
            static/images/icon-${size}.png
done
```

## Alternative: Use Online Generator

Visit https://www.pwabuilder.com/imageGenerator to generate all required sizes from a single image.
