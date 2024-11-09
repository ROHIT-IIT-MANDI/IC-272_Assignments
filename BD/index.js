const express = require('express');
const app = express();
const bodyParser = require('body-parser');


require("dotenv").config();
const PORT = process.env.PORT || 4000;


app.use(express.json());



//import routes for todo api
const todoRoutes = require("./routes/todos.js");
//mount the todo api routes
app.use("api/v1",todoRoutes);

//start server
app.listen(PORT,()=>{
    console.log(`Server is running on http://localhost:${PORT}/`);
});



