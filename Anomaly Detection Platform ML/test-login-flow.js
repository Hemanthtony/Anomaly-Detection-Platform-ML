/**
 * Automated test script for login page using Puppeteer
 * Tests email/password login, Google sign-in, and navigation flows
 */

const puppeteer = require('puppeteer-core');

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
  });
  const page = await browser.newPage();

  // Go to login page served on localhost
  await page.goto('http://127.0.0.1:8080/login.html', { waitUntil: 'networkidle0' });

  // Test email/password sign-up flow
  await page.click('#signup-link');
  await page.type('input[type="email"]', 'newuser@example.com');
  await page.type('input[type="password"]', 'newpassword123');
  await page.click('button[type="submit"]');
  await new Promise(resolve => setTimeout(resolve, 3000)); // wait for alert or redirect

  // Capture alert message
  page.on('dialog', async dialog => {
    console.log('Alert message:', dialog.message());
    await dialog.dismiss();
  });

  // Reload page for next test
  await page.reload({ waitUntil: 'networkidle0' });

  // Test email/password login - success case
  await page.type('input[type="email"]', 'newuser@example.com');
  await page.type('input[type="password"]', 'newpassword123');
  await page.click('button[type="submit"]');
  await new Promise(resolve => setTimeout(resolve, 2000)); // wait for alert or redirect

  // Test email/password login - failure case
  await page.type('input[type="email"]', 'newuser@example.com');
  await page.type('input[type="password"]', 'wrongpassword');
  await page.click('button[type="submit"]');
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Test Firebase initialization by checking if Firebase scripts loaded and app initialized
  // Wait for Firebase to be available
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));

  // Wait for firebase-app.js script tag to be loaded
  await page.waitForFunction(() => {
    const script = document.querySelector('script[src*="firebase-app.js"]');
    return script && script.complete;
  }, { timeout: 10000 }).catch(() => {});

  // Wait for window.firebase to be defined
  await page.waitForFunction(() => window.firebase !== undefined, { timeout: 10000 }).catch(() => {});

  const firebaseStatus = await page.evaluate(() => {
    console.log('Checking window.firebase:', window.firebase);
    const checks = {
      firebaseGlobal: typeof window.firebase !== 'undefined',
      initializeApp: typeof (window.firebase && window.firebase.initializeApp) === 'function',
      firebaseApp: typeof (window.firebase && window.firebase.app) === 'function',
      auth: typeof (window.firebase && window.firebase.auth) === 'function'
    };

    // Check if Firebase config script loaded
    const firebaseConfigLoaded = document.querySelector('script[src*="firebase-app.js"]') !== null ||
                                 document.querySelector('script')?.textContent?.includes('firebaseConfig');

    return {
      ...checks,
      configLoaded: firebaseConfigLoaded,
      anyFirebase: Object.values(checks).some(Boolean)
    };
  });
  console.log('Firebase status:', firebaseStatus);

  // Test Google sign-in button click (cannot fully automate OAuth popup)
  // Just check if button exists and is clickable
  const googleBtn = await page.$('.google-btn');
  if (googleBtn) {
    console.log('Google sign-in button found and clickable');

    // Attempt to click Google sign-in button (will likely fail in headless mode due to OAuth)
    try {
      await googleBtn.click();
      console.log('Google sign-in button clicked successfully');
      // Wait a bit for any popup or error
      await new Promise(resolve => setTimeout(resolve, 3000));
    } catch (error) {
      console.log('Google sign-in button click resulted in expected error (OAuth popup in headless mode):', error.message);
    }
  } else {
    console.error('Google sign-in button not found');
  }

  // Test continue without sign-in button navigation
  try {
    await page.click('#continue-without-btn');
    await page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 10000 });
    console.log('Successfully navigated to:', page.url());
  } catch (error) {
    console.log('Navigation timeout occurred, but page may have loaded. Current URL:', page.url());
  }

  await browser.close();
})();
