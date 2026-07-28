import {
  FiBookOpen,
  FiHeart,
  FiTruck,
  FiDollarSign,
} from "react-icons/fi";

export default function PromptCards() {
  return (
    <section className="prompt-section">

      <div className="prompt-card">
        <FiBookOpen className="card-icon" />

        <h3>शिक्षा</h3>

        <p>
          शिक्षा क्षेत्रमा कति बजेट विनियोजन गरिएको छ?
        </p>
      </div>

      <div className="prompt-card">
        <FiHeart className="card-icon" />

        <h3>स्वास्थ्य</h3>

        <p>
          स्वास्थ्य क्षेत्रमा कति बजेट छुट्याइएको छ?
        </p>
      </div>

      <div className="prompt-card">
        <FiTruck className="card-icon" />

        <h3>पूर्वाधार</h3>

        <p>
          पूर्वाधार विकासका लागि कति बजेट विनियोजन गरिएको छ?
        </p>
      </div>

      <div className="prompt-card">
        <FiDollarSign className="card-icon" />

        <h3>राजस्व</h3>

        <p>
          नेपालको प्रमुख राजस्वका स्रोतहरू के–के हुन्?
        </p>
      </div>

    </section>
  );
}