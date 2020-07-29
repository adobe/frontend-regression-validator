function getBgImgs (doc) {
 const srcChecker = /url\(\s*?['"]?\s*?(\S+?)\s*?["']?\s*?\)/i
 return Array.from(
   Array.from(doc.querySelectorAll('*'))
     .reduce((collection, node) => {
       let prop = window.getComputedStyle(node, null)
         .getPropertyValue('background-image')
       // match url(...)
       let match = srcChecker.exec(prop)
       if (match) {
         collection.add(node)
       }
       return collection
     }, new Set())
 )
}
var images = Array.from(document.querySelectorAll("img", "picture"))
var bg_matches = Array.from(getBgImgs(document))
for(i=0;i<bg_matches.length;i++){
    images.push(bg_matches[i])}

function offset(el) {

    var rect = el.getBoundingClientRect()
    var x = rect.x + pageXOffset
    var y = rect.y + pageYOffset
    var width = rect.width
    var height = rect.height

    return {x:x , y:y, width:width, height:height};
    }

var coordinates = []
for(var i =0;i<images.length;i++){
    coordinates.push(offset(images[i]));
    }

return coordinates