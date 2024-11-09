import "./App.css";
import { useState } from "react";
import data from './data.js'
import Tours from "./components/Tours"




function App() {

  const [tours,setTours] = useState(data)
   function removeTour(id){
    const newTours = tours.filter(tour => tour.id !==id);
    setTours(newTours);

   }

   if(tours.length()===0){
    return(
      <div>
        <button className = "btn" onClick={()=>{
          setTours(data)
        }}>
          Refresh
        </button>
      </div>
    );
  }


  return (
    <div>

      <Tours tour = {tours} removeTour = {removeTour}></Tours>

    </div>
  );
}

export default App;
