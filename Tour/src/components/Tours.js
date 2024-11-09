
import Card from "./Card.js"
import "./Tours.css"

function Tours({tour,removeTour}){

    


    return(
        <div>
            <div>
                <h2> Plan your Journey</h2>
            </div>

            <div className = "template"> 
                {
                    tour.map( (tour) => {
                        return <Card {...tour} removeTour={removeTour}>
                               
                            </Card>
                    })
                }   
            </div>

        </div>
    )
}

export default Tours;