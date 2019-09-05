function getQueryVariable(variable)
{
       var query = window.location.search.substring(1);
       var vars = query.split("&");
       for (var i=0;i<vars.length;i++) {
               var pair = vars[i].split("=");
               if(pair[0] == variable){return pair[1];}
       }
       return(false);
}

var global_response;
var current_id;
$(function(){
 arr = {"id": getQueryVariable("id")}
 $.ajax({
   type: "GET",
   url: "/api/result",
   data: arr,
   success: function(response) {
     global_response = response;
     $.each(response, function(i, item){
       var $x = "";
       var $overall_mask_div = parseFloat(item.ui_stats.mask_div.overall);
       var $overall_pixel_div = parseFloat(item.ui_stats.pixelwise_div.overall);
       if($overall_mask_div >= 0.3 || $overall_pixel_div >= 0.3) {
         $x += "<button id=\""+ i +"\" \"type=\"button\" class=\"btn btn-danger btn-sm btn-block\">"+ item.links.endpoint +"</button>";
       }
       else{
         $x += "<button id=\""+ i +"\" \"type=\"button\" class=\"btn btn-success btn-sm btn-block\">"+ item.links.endpoint +"</button>";
       }
       $("#buttons").append($x);
     });
   },
   error: function(response) {
     $("#wrong_id_alert").toggleClass("d-none");
   }
 });

});

$(document).on('click', 'button', function(){
     var $btn_id = this.id;
     $current_id = this.id;
     var item = global_response[$btn_id];

     var $progress_bar_mask_overall = "<div class=\"progress\">";
     $progress_bar_mask_overall += "<div class=\"progress-bar\" role=\"progressbar\" style=\"width: "+(parseFloat(item.ui_stats.mask_div.overall) * 100).toString()+"%;\" aria-valuenow="+(parseFloat(item.ui_stats.mask_div.overall) * 100).toString()+" aria-valuemin=\"0\" aria-valuemax=\"1\">Mask: " +item.ui_stats.mask_div.overall+"</div>";
     $progress_bar_mask_overall += "</div>";

     document.querySelectorAll("#overall_progress_mask")[0].innerHTML = $progress_bar_mask_overall;

     var $progress_bar_pixel_overall = "<div class=\"progress\">";
     $progress_bar_pixel_overall += "<div class=\"progress-bar\" role=\"progressbar\" style=\"width: "+(parseFloat(item.ui_stats.pixelwise_div.overall) * 100).toString()+"%;\" aria-valuenow="+(parseFloat(item.ui_stats.pixelwise_div.overall) * 100).toString()+" aria-valuemin=\"0\" aria-valuemax=\"1\">Content: " +item.ui_stats.pixelwise_div.overall+"</div>";
     $progress_bar_pixel_overall += "</div>";

     document.querySelectorAll("#overall_progress_pixel")[0].innerHTML = $progress_bar_pixel_overall;
     var $x = "";
     $("#current_card").remove();
     $x += "<div id=\"current_card\" class=\"card\">";

     $x += "<div class=\"card-body\">";
     $x += "<p class=\"card-text\">";

     $x += "<table class=\"table table-sm\">";
     $x += "<thead class=\"thead-light\">";
     $x += "<tr>";
     $x += "<th scope=\"col\" id=\"col1_h\">Divergence</th>";
     $x += "<th scope=\"col\" id=\"col2_h\">Images</th>";

     $x += "</tr>";
     $x += "</thead>";

     $x += "<tbody>";

     $x += "<tr>";
     $x += "<th scope=\"row\" id=\"r1c1\">";
     $x += "Mask"
     $x += "</th>";
     $x += "<td id=\"r1c2\">";
     $x += item.ui_stats.mask_div.images;
     $x += "</td>";

     $x += "</tr>";

     $x += "<tr>";
     $x += "<th scope=\"row\" id=\"r2c1\">";
     $x += "Content"
     $x += "</th>";
     $x += "<td id=\"r2c2\">";
     $x += item.ui_stats.pixelwise_div.images;
     $x += "</td>";
     $x += "</tr>";

     $x += "</tbody>";

     $x += "</table>";



     $x += "</p>";
     $x += "</div>";
     $x += "</div>";

     var $images_cocoen = "";
     $images_cocoen += "<img src=\"/static/" + item.links.baseline_assets + $btn_id + ".png\" class=\"img-thumbnail\" style=\"max-height:100%; width:auto; height:auto\">";
     $images_cocoen += "<img src=\"/static/" + item.links.updated_assets + $btn_id + ".png\" class=\"img-thumbnail\" style=\"max-height:100%; width:auto; height:auto\">";

     $("#table").append($x);
     document.querySelectorAll("#images_cocoen")[0].innerHTML = $images_cocoen;
     $("#images_msk").click();



});

$("#buttons_msk").click(function(){
    document.querySelectorAll("th")[1].innerText = "Buttons";
    document.querySelectorAll("td")[0].innerText = global_response[$current_id].ui_stats.mask_div.buttons;
    document.querySelectorAll("td")[1].innerText = global_response[$current_id].ui_stats.pixelwise_div.buttons;

    document.querySelectorAll("img")[1].src = "/static/" + global_response[$current_id].links.baseline_assets + "buttons_" + $current_id + ".png";
    document.querySelectorAll("img")[2].src = "/static/" + global_response[$current_id].links.updated_assets + "buttons_" + $current_id + ".png";
});

$("#sections_msk").click(function(){
    document.querySelectorAll("th")[1].innerText = "Sections";
    document.querySelectorAll("td")[0].innerText = global_response[$current_id].ui_stats.mask_div.section;
    document.querySelectorAll("td")[1].innerText = global_response[$current_id].ui_stats.pixelwise_div.section;

    document.querySelectorAll("img")[1].src = "/static/" + global_response[$current_id].links.baseline_assets + "section_" + $current_id + ".png";
    document.querySelectorAll("img")[2].src = "/static/" + global_response[$current_id].links.updated_assets + "section_" + $current_id + ".png";
});

$("#images_msk").click(function(){
    document.querySelectorAll("th")[1].innerText = "Images";
    document.querySelectorAll("td")[0].innerText = global_response[$current_id].ui_stats.mask_div.images;
    document.querySelectorAll("td")[1].innerText = global_response[$current_id].ui_stats.pixelwise_div.images;

    document.querySelectorAll("img")[1].src = "/static/" + global_response[$current_id].links.baseline_assets + "images_" + $current_id + ".png";
    document.querySelectorAll("img")[2].src = "/static/" + global_response[$current_id].links.updated_assets + "images_" + $current_id + ".png";
});

$("#forms_msk").click(function(){
    document.querySelectorAll("th")[1].innerText = "Forms";
    document.querySelectorAll("td")[0].innerText = global_response[$current_id].ui_stats.mask_div.forms;
    document.querySelectorAll("td")[1].innerText = global_response[$current_id].ui_stats.pixelwise_div.forms;

    document.querySelectorAll("img")[1].src = "/static/" + global_response[$current_id].links.baseline_assets + "forms_" + $current_id + ".png";
    document.querySelectorAll("img")[2].src = "/static/" + global_response[$current_id].links.updated_assets + "forms_" + $current_id + ".png";
});

$("#textblocks_msk").click(function(){
    document.querySelectorAll("th")[1].innerText = "Textblocks";
    document.querySelectorAll("td")[0].innerText = global_response[$current_id].ui_stats.mask_div.textblock;
    document.querySelectorAll("td")[1].innerText = global_response[$current_id].ui_stats.pixelwise_div.textblock;

    document.querySelectorAll("img")[1].src = "/static/" + global_response[$current_id].links.baseline_assets + "textblock_" + $current_id + ".png";
    document.querySelectorAll("img")[2].src = "/static/" + global_response[$current_id].links.updated_assets + "textblock_" + $current_id + ".png";
});
