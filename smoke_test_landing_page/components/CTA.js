// smoke_test_landing_page/components/CTA.js
import React, { useState } from 'react';

export default function CTA() {
  const [email, setEmail] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    // In a real app, you'd send this email to a backend or email service
    console.log('Email submitted:', email);
    setSubmitted(true);
    // Reset form or show thank you message
    setTimeout(() => {
      setEmail('');
      setSubmitted(false);
    }, 3000);
  };

  if (submitted) {
    return <p>Thanks for signing up! We'll be in touch.</p>;
  }

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: '30px' }}>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Enter your email"
        required
        style={{ padding: '10px', marginRight: '10px', minWidth: '250px' }}
      />
      <button type="submit" style={{ padding: '10px 15px' }}>
        Sign Up
      </button>
    </form>
  );
}
