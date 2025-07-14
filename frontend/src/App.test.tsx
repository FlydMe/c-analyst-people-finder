import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';

describe('VC Analyst Search Tool', () => {
  it('performs a full search flow in under 5 minutes', async () => {
    render(<App />);
    // Fill out the form
    fireEvent.change(screen.getByPlaceholderText(/e\.g\., Google/i), { target: { value: 'Stripe' } });
    fireEvent.change(screen.getByPlaceholderText(/e\.g\., Senior Engineer, Product Manager/i), { target: { value: 'Software Engineer' } });
    fireEvent.change(screen.getByLabelText(/Seniority/i), { target: { value: 'Senior' } });
    fireEvent.change(screen.getByLabelText(/Quit Window/i), { target: { value: '6 months' } });
    fireEvent.change(screen.getByPlaceholderText(/e\.g\., India, San Francisco/i), { target: { value: 'San Francisco' } });
    fireEvent.change(screen.getByPlaceholderText(/e\.g\., AI, ML, cloud/i), { target: { value: 'Python' } });
    fireEvent.change(screen.getByPlaceholderText(/e\.g\., intern, junior/i), { target: { value: 'intern' } });

    // Generate queries
    fireEvent.click(screen.getByText(/Generate X-ray Queries/i));
    await waitFor(() => expect(screen.getByText(/🧾 Generated X-ray Queries/i)).toBeInTheDocument(), { timeout: 60000 });

    // Execute search
    fireEvent.click(screen.getByText(/Execute Search/i));
    const start = Date.now();
    await waitFor(() => {
      expect(screen.getByText(/Search Results/i)).toBeInTheDocument();
      expect(screen.getByText(/potential candidates/i)).toBeInTheDocument();
    }, { timeout: 300000 }); // 5 minutes
    const elapsed = (Date.now() - start) / 1000;
    // Debug output
    if (elapsed > 300) {
      // eslint-disable-next-line no-console
      console.error('Search took too long:', elapsed, 'seconds');
    }
    expect(elapsed).toBeLessThan(300);
  });
}); 