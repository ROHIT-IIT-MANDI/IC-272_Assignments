//import the model
const Todo = require("../models/Todo.js");

//define route handler
exports.createTodo = async(req,res) => {
    try{
        //extract title and description from the request body
        const {title,description} = req.body;
        //create a new Todo Object and insert in db
        const response = await Todo.create({title,description});
        //send a json response with a sucuess flag
        res.status(200).json(
            {
                sucess:true,
                data:response,
                message:'Entry Created Sucessfully'
            }
        )
    }
    catch(err){

    }
}
