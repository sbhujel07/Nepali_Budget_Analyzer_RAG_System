import {
  FiMap,
  FiHeart,
  FiSun,
  FiDollarSign,
} from "react-icons/fi";

interface PromptCardsProps {
  onPromptSelect: (prompt: string) => void;
}

export default function PromptCards({
  onPromptSelect,
}: PromptCardsProps) {
  return (
    <section className="prompt-section">

      <div
        className="prompt-card"
        onClick={() =>{
          onPromptSelect(
            "नेपालको प्रति व्यक्ति राष्ट्रिय आय कति पुगेको छ?"
          )
        }}
      >
        <FiDollarSign className="card-icon" />

        <h3>अर्थतन्त्र</h3>

        <p>
          नेपालको प्रति व्यक्ति राष्ट्रिय आय कति पुगेको छ?
        </p>
      </div>

      <div
        className="prompt-card"
        onClick={() =>
          onPromptSelect(
            "स्वास्थ्य क्षेत्रमा कति बजेट छुट्याइएको छ?"
          )
        }
      >
        <FiHeart className="card-icon" />

        <h3>स्वास्थ्य</h3>

        <p>
          स्वास्थ्य क्षेत्रमा कति बजेट छुट्याइएको छ?
        </p>
      </div>

      <div
        className="prompt-card"
        onClick={() =>
          onPromptSelect(
            "पर्यटन क्षेत्रका लागि कति बजेट विनियोजन गरिएको छ?"
          )
        }
      >
        <FiMap className="card-icon" />

        <h3>पर्यटन </h3>

        <p>
          पर्यटन क्षेत्रका लागि कति बजेट विनियोजन गरिएको छ?
        </p>
      </div>

      <div
        className="prompt-card"
        onClick={() =>
          onPromptSelect(
            "कृषि क्षेत्रको विकासका लागि सरकारले के योजना अघि सारेको छ?"
          )
        }
      >
        <FiSun className="card-icon" />

        <h3>कृषि</h3>

        <p>
          कृषि क्षेत्रको विकासका लागि सरकारले के योजना अघि सारेको छ?
        </p>
      </div>

    </section>
  );
}