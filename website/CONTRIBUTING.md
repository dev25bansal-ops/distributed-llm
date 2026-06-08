# Contributing to DistLLM Website

Thank you for your interest in contributing to the DistLLM website! This guide will help you get started.

## Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/distributed-llm.git`
3. **Install dependencies**: `cd website && npm install`
4. **Start dev server**: `npm run dev`
5. **Open** http://localhost:3000

## Development Workflow

### Branch Naming
- `feature/description` — New features
- `fix/description` — Bug fixes
- `docs/description` — Documentation updates
- `refactor/description` — Code refactoring

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add dark mode toggle
fix: correct calculator electricity cost
docs: update API reference
style: format code with prettier
```

### Code Style
- **JavaScript**: ES modules, no framework
- **CSS**: CSS custom properties, BEM-like naming
- **HTML**: Semantic, accessible

Run the linter:
```bash
npm run lint
```

Format code:
```bash
npm run format
```

## Project Structure

```
website/
├── css/                  # Stylesheets
│   ├── base.css         # Reset, typography, variables
│   ├── layout.css       # Grid, header, footer
│   ├── components.css   # Cards, buttons, forms
│   ├── animations.css   # Scroll animations
│   ├── themes.css       # Dark/light themes
│   ├── advanced.css     # Complex components
│   └── print.css        # Print styles
├── js/                   # JavaScript modules
│   ├── main.js          # Entry point, module loader
│   ├── theme.js         # Theme toggle
│   ├── calculator.js    # Savings calculator
│   └── ...              # Feature modules
├── tests/                # Test files
│   ├── integration.spec.js
│   ├── accessibility.spec.js
│   └── ...
└── *.html                # Pages
```

## Adding a New Page

1. Create `new-page.html` in the root
2. Add to `sitemap.xml`
3. Add navigation link in `layout.css` header
4. Add BreadcrumbList structured data
5. Add OpenGraph meta tags

## Adding a New Component

1. **HTML**: Add semantic markup to the page
2. **CSS**: Add styles to `advanced.css` using CSS variables
3. **JS**: Create module in `js/` directory
4. **Import**: Add to `main.js` lazy loading
5. **Test**: Add to integration tests

Example:
```js
// js/new-component.js
export function initNewComponent() {
    const container = document.getElementById('newComponent');
    if (!container) return;
    
    container.innerHTML = `
        <div class="new-component">
            <!-- Component content -->
        </div>
    `;
}
```

## Testing

### Unit Tests
```bash
npm run test:unit
```

### E2E Tests
```bash
npm run test:e2e
```

### Run All Tests
```bash
npm test
```

## Design System

### Colors
Use CSS custom properties:
```css
/* Dark theme (default) */
--bg: #050505;
--surface: #0c0c0c;
--card: #111113;
--border: #1c1c1e;
--text: #ededed;
--green: #00e676;

/* Light theme */
[data-theme="light"] {
    --bg: #fafafa;
    --surface: #f5f5f5;
    --card: #ffffff;
    --border: #e5e5e5;
    --text: #111111;
}
```

### Typography
```css
--font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', monospace;
```

### Spacing
Use consistent spacing:
```css
--space-xs: 4px;
--space-sm: 8px;
--space-md: 16px;
--space-lg: 24px;
--space-xl: 32px;
--space-2xl: 48px;
```

## Accessibility

- Use semantic HTML (`<nav>`, `<main>`, `<section>`, `<article>`)
- Add `aria-label` to interactive elements
- Ensure keyboard navigation works
- Maintain color contrast (4.5:1 minimum)
- Test with screen readers

## Performance

- Lazy load below-the-fold content
- Use `loading="lazy"` for images
- Minimize inline styles (use CSS classes)
- Use CSS variables for theming
- Test with Lighthouse

## Pull Request Process

1. **Create** a feature branch
2. **Make** your changes
3. **Test** locally: `npm test`
4. **Lint** your code: `npm run lint`
5. **Commit** with conventional commits
6. **Push** to your fork
7. **Create** a pull request

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor

## Testing
- [ ] Unit tests pass
- [ ] E2E tests pass
- [ ] Manual testing done

## Screenshots
(If applicable)
```

## Design Files

- **Figma**: [DistLLM Design System](https://figma.com/distllm)
- **Icons**: SVG sprites in `index.html`
- **Colors**: CSS variables in `themes.css`

## Component Inventory

| Component | File | Description |
|-----------|------|-------------|
| Hero | `index.html` | Homepage hero section |
| Calculator | `calculator.js` | Savings calculator |
| GPU Checker | `gpu-checker.js` | GPU compatibility |
| Model Explorer | `model-explorer.js` | Model browser |
| Deploy Wizard | `deploy-wizard.js` | Deployment guide |
| Chat Demo | `chat-demo.js` | Interactive demo |
| Benchmark Dashboard | `bench-dashboard.js` | Performance charts |
| Community Hub | `community-hub.js` | Community features |
| AI Chatbot | `ai-chatbot.js` | AI assistant |

## Getting Help

- **Discord**: [#website channel](https://discord.gg/distllm)
- **GitHub Issues**: [Website issues](https://github.com/distributed-llm/distributed-llm/labels/website)
- **Discussions**: [GitHub Discussions](https://github.com/distributed-llm/distributed-llm/discussions)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
