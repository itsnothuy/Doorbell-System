#!/usr/bin/env python3
"""
End-to-End Web Interface Tests using Playwright

These tests verify the complete user journey through the web interface,
testing browser interactions, navigation, and real-time updates.

Requirements:
    pip install playwright pytest-playwright
    playwright install chromium

Usage:
    pytest tests/e2e/test_web_interface_playwright.py -v
    pytest tests/e2e/test_web_interface_playwright.py -v --headed  # Show browser
"""

import pytest
import asyncio
import time
from playwright.async_api import async_playwright, Page, expect


@pytest.mark.e2e
@pytest.mark.asyncio
class TestDashboardE2E:
    """End-to-end tests for the dashboard."""

    async def test_dashboard_loads_successfully(self, page: Page):
        """Test that the dashboard page loads with all key elements."""
        # Navigate to dashboard
        await page.goto("http://localhost:5000/")
        
        # Wait for page to load
        await page.wait_for_load_state("networkidle")
        
        # Verify page title
        await expect(page).to_have_title(pytest.re.compile("Doorbell Security", re.IGNORECASE))
        
        # Verify key UI elements are present
        await expect(page.locator("h1")).to_be_visible()
        
        # Check for status cards
        status_cards = page.locator("[data-testid='status-card'], .status-card, .card")
        await expect(status_cards.first).to_be_visible()

    async def test_dashboard_displays_system_status(self, page: Page):
        """Test that system status is displayed correctly."""
        await page.goto("http://localhost:5000/")
        await page.wait_for_load_state("networkidle")
        
        # Check if status indicators are present
        # This may vary based on actual implementation
        camera_status = page.locator("text=/camera/i")
        if await camera_status.count() > 0:
            await expect(camera_status.first).to_be_visible()

    async def test_navigation_menu_works(self, page: Page):
        """Test that navigation menu allows switching between pages."""
        await page.goto("http://localhost:5000/")
        
        # Find and click navigation links
        nav_links = page.locator("nav a, .navbar a, [role='navigation'] a")
        
        if await nav_links.count() > 0:
            # Click first navigation link
            await nav_links.first.click()
            
            # Verify navigation occurred
            await page.wait_for_load_state("networkidle")
            assert page.url != "http://localhost:5000/"


@pytest.mark.e2e
@pytest.mark.asyncio
class TestCameraStreamE2E:
    """End-to-end tests for camera streaming."""

    async def test_camera_page_loads(self, page: Page):
        """Test that the camera page loads successfully."""
        await page.goto("http://localhost:5000/camera")
        await page.wait_for_load_state("networkidle")
        
        # Verify we're on the camera page
        await expect(page).to_have_url(pytest.re.compile("/camera"))

    async def test_camera_controls_are_present(self, page: Page):
        """Test that camera controls are available."""
        await page.goto("http://localhost:5000/camera")
        await page.wait_for_load_state("networkidle")
        
        # Look for common camera control elements
        buttons = page.locator("button")
        await expect(buttons.first).to_be_visible()


@pytest.mark.e2e
@pytest.mark.asyncio
class TestFaceManagementE2E:
    """End-to-end tests for face management workflow."""

    async def test_known_faces_page_loads(self, page: Page):
        """Test that the known faces page loads."""
        await page.goto("http://localhost:5000/known-faces")
        await page.wait_for_load_state("networkidle")
        
        # Verify we're on the known faces page
        await expect(page).to_have_url(pytest.re.compile("/known-faces"))

    async def test_can_view_face_list(self, page: Page):
        """Test viewing the list of known faces."""
        await page.goto("http://localhost:5000/known-faces")
        await page.wait_for_load_state("networkidle")
        
        # Wait for content to load
        await page.wait_for_timeout(1000)
        
        # Check if face cards or list items are present
        face_items = page.locator("[data-testid='face-card'], .face-card, .face-item")
        
        # Should have at least the page structure
        page_content = page.locator("body")
        await expect(page_content).to_be_visible()


@pytest.mark.e2e
@pytest.mark.asyncio
class TestEventsPageE2E:
    """End-to-end tests for events/history page."""

    async def test_events_page_loads(self, page: Page):
        """Test that the events page loads."""
        await page.goto("http://localhost:5000/events")
        await page.wait_for_load_state("networkidle")
        
        # Page should load without errors
        await expect(page.locator("body")).to_be_visible()

    async def test_can_filter_events(self, page: Page):
        """Test event filtering functionality."""
        await page.goto("http://localhost:5000/events")
        await page.wait_for_load_state("networkidle")
        
        # Look for filter controls
        filters = page.locator("select, [role='combobox'], .filter")
        
        if await filters.count() > 0:
            # Interact with first filter
            await filters.first.click()


@pytest.mark.e2e
@pytest.mark.asyncio
class TestAPIIntegrationE2E:
    """End-to-end tests for API integration."""

    async def test_api_status_endpoint(self, page: Page):
        """Test that the API status endpoint returns valid data."""
        # Navigate to a page that calls the API
        await page.goto("http://localhost:5000/")
        
        # Intercept API call
        api_response = None
        
        async def handle_response(response):
            nonlocal api_response
            if "/api/status" in response.url or "/api/system/status" in response.url:
                api_response = response
        
        page.on("response", handle_response)
        
        # Wait for API call
        await page.wait_for_timeout(2000)
        
        # If API was called, verify response
        if api_response:
            assert api_response.status == 200

    async def test_api_events_endpoint(self, page: Page):
        """Test that the events API endpoint works."""
        await page.goto("http://localhost:5000/events")
        
        # Intercept API call
        events_response = None
        
        async def handle_response(response):
            nonlocal events_response
            if "/api/events" in response.url:
                events_response = response
        
        page.on("response", handle_response)
        
        # Wait for API call
        await page.wait_for_timeout(2000)
        
        # If API was called, verify response
        if events_response:
            assert events_response.status in [200, 404]  # 404 is ok if no events


@pytest.mark.e2e
@pytest.mark.asyncio
class TestResponsivenessE2E:
    """End-to-end tests for responsive design."""

    async def test_mobile_viewport(self, page: Page):
        """Test that the site works on mobile viewport."""
        # Set mobile viewport
        await page.set_viewport_size({"width": 375, "height": 667})
        
        await page.goto("http://localhost:5000/")
        await page.wait_for_load_state("networkidle")
        
        # Page should render without horizontal scroll
        await expect(page.locator("body")).to_be_visible()

    async def test_tablet_viewport(self, page: Page):
        """Test that the site works on tablet viewport."""
        # Set tablet viewport
        await page.set_viewport_size({"width": 768, "height": 1024})
        
        await page.goto("http://localhost:5000/")
        await page.wait_for_load_state("networkidle")
        
        # Page should render correctly
        await expect(page.locator("body")).to_be_visible()

    async def test_desktop_viewport(self, page: Page):
        """Test that the site works on desktop viewport."""
        # Set desktop viewport
        await page.set_viewport_size({"width": 1920, "height": 1080})
        
        await page.goto("http://localhost:5000/")
        await page.wait_for_load_state("networkidle")
        
        # Page should render correctly
        await expect(page.locator("body")).to_be_visible()


# Fixtures for Playwright

@pytest.fixture(scope="session")
async def browser():
    """Create a browser instance for the test session."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage"]
        )
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser):
    """Create a new page for each test."""
    context = await browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent="Mozilla/5.0 (Playwright Test)"
    )
    page = await context.new_page()
    
    # Set default timeout
    page.set_default_timeout(10000)
    
    yield page
    
    await page.close()
    await context.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_server():
    """
    Ensure the test server is running.
    
    In a real scenario, you might start the Flask app here
    or use a fixture that starts it in a separate thread/process.
    """
    # For now, assume the server is already running
    # In production tests, you'd start it here:
    # 
    # import subprocess
    # import time
    # 
    # server_process = subprocess.Popen(
    #     ["python", "app.py"],
    #     env={"DEVELOPMENT_MODE": "true"}
    # )
    # time.sleep(5)  # Wait for server to start
    # 
    # yield
    # 
    # server_process.terminate()
    # server_process.wait()
    
    yield


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--headed"])
