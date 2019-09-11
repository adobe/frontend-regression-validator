function offset(el) {

    var rect = el.getBoundingClientRect()
    var x = rect.x + pageXOffset
    var y = rect.y + pageYOffset
    var width = rect.width
    var height = rect.height

    return {x:x , y:y, width:width, height:height};
    }

var coordinates = []
var matches = Array.from(document.querySelectorAll("argumentName"))

for(var i =0;i<matches.length;i++){
    coordinates.push(offset(matches[i]));
    }

return coordinates
