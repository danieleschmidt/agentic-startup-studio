// smoke_test_landing_page/pages/index.js
import Head from 'next/head';
import CTA from '../components/CTA';

export default function HomePage() {
  return (
    <div>
      <Head>
        <title>Our Amazing Startup Idea</title>
        <meta name="description" content="Sign up to learn more!" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main style={{ textAlign: 'center', marginTop: '50px' }}>
        <h1>Welcome to [Startup Name Placeholder]!</h1>
        <p>
          We are solving [Problem Placeholder] with [Solution Placeholder].
          Sign up below to get early access and exclusive updates!
        </p>
        <CTA />
      </main>

      <footer style={{ textAlign: 'center', marginTop: '50px', padding: '20px', borderTop: '1px solid #eaeaea' }}>
        <p>&copy; {new Date().getFullYear()} [Startup Name Placeholder]</p>
      </footer>
    </div>
  );
}
