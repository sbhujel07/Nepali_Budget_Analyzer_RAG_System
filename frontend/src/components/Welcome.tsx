import { FiTrendingUp } from "react-icons/fi";

export default function Welcome() {
  return (
    <section className="welcome-section">

      <div className="welcome-icon">
        <FiTrendingUp size={48} />
      </div>

      <h1>Welcome to Nepal Budget AI</h1>

      <p>
        Ask questions about Nepal's annual budget, allocations,
        revenue, expenditure, and government financial policies.
      </p>

    </section>
  );
}