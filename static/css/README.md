# Frontend CSS Architecture

## Overview

The Doorbell Security System frontend uses a modern, modular CSS architecture based on a design system approach. The CSS is organized into three main files that work together to provide a comprehensive styling solution.

## File Structure

```
static/css/
├── design-system.css   # Core design tokens and utilities
├── components.css      # Reusable UI components
└── layout.css          # Layout and page structure
```

## Design System (`design-system.css`)

The design system file contains all the foundational design tokens used throughout the application.

### CSS Custom Properties

All design tokens are defined as CSS custom properties (variables) for consistency and easy theming:

```css
:root {
  /* Colors */
  --primary-blue: #2563eb;
  --security-green: #059669;
  --warning-amber: #d97706;
  --danger-red: #dc2626;
  
  /* Typography */
  --font-family-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  
  /* Spacing */
  --space-4: 1rem;
  --space-6: 1.5rem;
  
  /* Transitions */
  --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
}
```

### Utility Classes

The design system provides utility classes for rapid development:

- **Spacing**: `.m-4`, `.p-6`, `.mt-2`, `.mb-4`
- **Typography**: `.text-lg`, `.font-bold`, `.text-center`
- **Display**: `.flex`, `.grid`, `.hidden`, `.block`
- **Colors**: `.text-primary`, `.bg-success`, `.text-gray-700`
- **Borders**: `.rounded-lg`, `.rounded-full`
- **Shadows**: `.shadow-md`, `.shadow-xl`

### Animations

Pre-defined animations for common UI interactions:

- **spin**: For loading spinners
- **pulse**: For status indicators
- **slideInRight**: For notifications
- **fadeIn**: For content reveals

## Components (`components.css`)

Reusable UI components with consistent styling.

### Button Component

```html
<!-- Primary Button -->
<button class="btn btn-primary">Click Me</button>

<!-- Success Button -->
<button class="btn btn-success">Success</button>

<!-- Danger Button -->
<button class="btn btn-danger">Delete</button>

<!-- Small Button -->
<button class="btn btn-sm btn-secondary">Small</button>

<!-- Loading State -->
<button class="btn btn-primary loading">Processing...</button>
```

### Card Component

```html
<div class="card">
  <div class="card-header">
    <h2 class="card-title">
      <iconify-icon icon="icon-name"></iconify-icon>
      Card Title
    </h2>
    <a href="#" class="card-action">View All</a>
  </div>
  <div class="card-content">
    <!-- Card content here -->
  </div>
</div>
```

### Status Indicator

```html
<div class="status-indicator" data-status="online">
  <span class="status-dot"></span>
  <span class="status-text">System Online</span>
</div>

<!-- Statuses: online, offline, warning -->
```

### Notification/Toast

```html
<div class="notification success show">
  <div class="notification-icon">✓</div>
  <div class="notification-content">
    <div class="notification-title">Success!</div>
    <div class="notification-message">Operation completed successfully.</div>
  </div>
  <button class="notification-close">×</button>
</div>

<!-- Types: success, error, warning, info -->
```

### Alert Component

```html
<div class="alert alert-success">
  <div class="alert-icon">✓</div>
  <div class="alert-content">
    <div class="alert-title">Success</div>
    <p>Your changes have been saved.</p>
  </div>
</div>

<!-- Types: alert-success, alert-warning, alert-error, alert-info -->
```

### Form Components

```html
<div class="form-group">
  <label class="form-label" for="input-id">Label</label>
  <input type="text" class="form-input" id="input-id" placeholder="Enter text">
</div>

<div class="form-group">
  <label class="form-label" for="select-id">Select</label>
  <select class="form-select" id="select-id">
    <option>Option 1</option>
    <option>Option 2</option>
  </select>
</div>
```

### Loading Components

```html
<!-- Spinner -->
<div class="spinner"></div>
<div class="spinner spinner-sm"></div>
<div class="spinner spinner-lg"></div>

<!-- Skeleton Loader -->
<div class="skeleton skeleton-text"></div>
<div class="skeleton skeleton-title"></div>
<div class="skeleton skeleton-avatar"></div>
```

### Badge Component

```html
<span class="badge badge-success">Active</span>
<span class="badge badge-warning">Pending</span>
<span class="badge badge-danger">Error</span>
<span class="badge badge-info">Info</span>
```

## Layout (`layout.css`)

Page structure and responsive design.

### Dashboard Layout

```html
<div class="dashboard-container">
  <aside class="sidebar">
    <!-- Sidebar content -->
  </aside>
  
  <main class="main-content">
    <header class="content-header">
      <h1 class="page-title">Page Title</h1>
      <div class="header-actions">
        <!-- Action buttons -->
      </div>
    </header>
    
    <section class="dashboard-grid">
      <!-- Cards and content -->
    </section>
  </main>
</div>
```

### Responsive Grid

The dashboard grid automatically adapts to different screen sizes:

- **Mobile (< 768px)**: 1 column
- **Tablet (768px - 1023px)**: 2 columns
- **Desktop (≥ 1024px)**: 3 columns

```html
<section class="dashboard-grid">
  <div class="card">Card 1</div>
  <div class="card">Card 2</div>
  <div class="card large">Large Card (spans 2 columns on desktop)</div>
  <div class="card full-width">Full Width Card (spans all columns)</div>
</section>
```

### Mobile Navigation

On mobile devices (≤ 768px), the sidebar becomes a slide-out menu:

```html
<!-- Mobile Toggle Button -->
<button id="mobile-nav-toggle" class="mobile-nav-toggle">
  <iconify-icon icon="material-symbols:menu"></iconify-icon>
</button>

<!-- Sidebar Overlay -->
<div id="sidebar-overlay" class="sidebar-overlay"></div>

<!-- Sidebar (with .open class when active) -->
<aside class="sidebar">
  <!-- Navigation content -->
</aside>
```

### Specialized Layouts

#### Status Grid
```html
<div class="status-grid">
  <div class="status-item">
    <span class="status-label">Label</span>
    <span class="status-value" data-status="online">Value</span>
  </div>
</div>
```

#### Known Faces Grid
```html
<div class="known-faces-grid">
  <div class="known-face-card">
    <img src="face.jpg" alt="Person name">
    <span class="name">John Doe</span>
    <button class="btn btn-danger btn-sm">Remove</button>
  </div>
</div>
```

#### Event List
```html
<div class="event-list">
  <div class="event-item known">
    <strong>Event Title</strong>
    <span class="event-time">2024-01-15 14:32:15</span>
  </div>
</div>

<!-- Event types: known, unknown, alert -->
```

## Responsive Breakpoints

```css
/* Mobile First Approach */
/* Base styles: Mobile (< 640px) */

@media (min-width: 640px)  { /* Small tablets */ }
@media (min-width: 768px)  { /* Tablets */ }
@media (min-width: 1024px) { /* Desktop */ }
@media (min-width: 1280px) { /* Large desktop */ }
@media (min-width: 1536px) { /* Extra large desktop */ }
```

## Accessibility Features

### Focus States
All interactive elements have visible focus states for keyboard navigation:
```css
*:focus-visible {
  outline: 2px solid var(--primary-blue);
  outline-offset: 2px;
}
```

### Screen Reader Support
```html
<!-- Screen reader only text -->
<span class="sr-only">Descriptive text for screen readers</span>

<!-- ARIA labels -->
<button aria-label="Close notification">×</button>
```

### Reduced Motion
Respects user's motion preferences:
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

### High Contrast Support
```css
@media (prefers-contrast: high) {
  .card {
    border: 2px solid var(--gray-900);
  }
}
```

## Best Practices

### 1. Use Design Tokens
Always use CSS custom properties instead of hardcoded values:
```css
/* Good */
color: var(--primary-blue);
padding: var(--space-4);

/* Avoid */
color: #2563eb;
padding: 16px;
```

### 2. Use Utility Classes
Leverage utility classes for common styling needs:
```html
<!-- Good -->
<div class="flex items-center gap-4">

<!-- Avoid -->
<div style="display: flex; align-items: center; gap: 1rem;">
```

### 3. Compose Components
Build complex UIs by composing existing components:
```html
<div class="card">
  <div class="card-header">
    <h2 class="card-title">Title</h2>
  </div>
  <div class="card-content">
    <div class="status-grid">
      <!-- Status items -->
    </div>
    <button class="btn btn-primary">Action</button>
  </div>
</div>
```

### 4. Mobile First
Write mobile styles first, then add desktop enhancements:
```css
/* Mobile styles (default) */
.dashboard-grid {
  grid-template-columns: 1fr;
}

/* Desktop enhancement */
@media (min-width: 1024px) {
  .dashboard-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}
```

## Color Palette Reference

### Primary Colors
- **Blue**: `--primary-blue` (#2563eb) - Primary actions, links
- **Blue Light**: `--primary-blue-light` (#3b82f6) - Hover states
- **Blue Dark**: `--primary-blue-dark` (#1d4ed8) - Active states

### Semantic Colors
- **Success/Security**: `--security-green` (#059669) - Success states, known visitors
- **Warning**: `--warning-amber` (#d97706) - Warnings, unknown visitors
- **Danger**: `--danger-red` (#dc2626) - Errors, delete actions, alerts
- **Info**: `--info-cyan` (#0891b2) - Information, notifications

### Status Colors
- **Online**: `--status-online` (#10b981) - System online, active
- **Offline**: `--status-offline` (#ef4444) - System offline, inactive
- **Warning**: `--status-warning` (#f59e0b) - Needs attention
- **Unknown**: `--status-unknown` (#6b7280) - Unknown state

### Neutral Colors
- **Gray 50-900**: Various shades for backgrounds, text, borders

## Typography Scale

- **xs**: 12px (0.75rem) - Small labels, captions
- **sm**: 14px (0.875rem) - Secondary text, descriptions
- **base**: 16px (1rem) - Body text (default)
- **lg**: 18px (1.125rem) - Emphasized text
- **xl**: 20px (1.25rem) - Small headings
- **2xl**: 24px (1.5rem) - Card headings
- **3xl**: 30px (1.875rem) - Page titles
- **4xl**: 36px (2.25rem) - Hero text

## Spacing Scale

- **1**: 4px (0.25rem) - Tight spacing
- **2**: 8px (0.5rem) - Small gaps
- **3**: 12px (0.75rem) - Medium-small gaps
- **4**: 16px (1rem) - Standard spacing
- **5**: 20px (1.25rem) - Medium spacing
- **6**: 24px (1.5rem) - Large spacing
- **8**: 32px (2rem) - Extra large spacing
- **10**: 40px (2.5rem) - Section spacing
- **12**: 48px (3rem) - Large section spacing
- **16**: 64px (4rem) - Hero spacing

## Browser Support

The CSS is designed to work in all modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

### Fallbacks
Progressive enhancement ensures graceful degradation:
- CSS Grid with flexbox fallback
- Custom properties with fallback values
- Modern features with @supports queries

## Performance Considerations

1. **Critical CSS**: Most important styles are in design-system.css
2. **Lazy Loading**: Components.css and layout.css can be lazy loaded
3. **Caching**: All CSS files are cacheable with service worker
4. **Minimal Specificity**: Low specificity for better performance
5. **No Unused Styles**: All styles are actively used in the application

## Future Enhancements

Potential improvements for future iterations:
- Dark mode support with additional color tokens
- Theme customization system
- CSS-in-JS migration option
- Additional component variants
- Animation library expansion
- Grid system enhancements
