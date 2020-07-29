$(function(){

 $.ajax({
     type: "GET",
     url: "/api/ids",
     dataType: 'json', // added data type
     success : function (response) {
       $.each(response, function(i, item) {
         var $x = "";

         $x += "<div class=\"card\">";
         $x += "<div class=\"card-header\" id=\"heading" + i + "\">";
         $x += "<h5 class=\"mb-0\">";
         if(item.status === "Done"){
            $x += "<button class=\"btn btn-success\" type=\"button\" data-toggle=\"collapse\" data-target=\"#collapse"+i+"\" aria-expanded=\"true\" aria-controls=\"collapse"+i+"\">";

         }
         else if(item.status === "In progress") {
           $x += "<button class=\"btn btn-info\" type=\"button\" data-toggle=\"collapse\" data-target=\"#collapse"+i+"\" aria-expanded=\"true\" aria-controls=\"collapse"+i+"\">";
         }
         else {
            $x += "<button class=\"btn btn-danger\" type=\"button\" data-toggle=\"collapse\" data-target=\"#collapse"+i+"\" aria-expanded=\"true\" aria-controls=\"collapse"+i+"\">";

         }
         $x += i;
         $x += "</button>";
         if(item.status === "Done") {
           $x += "<a href=\"/static/results.html?id="+ i +" \" class=\"btn btn-primary active float-right\" role=\"button\">View results</a>";
         }
         else {
           $x += "<a href=\"#\" class=\"btn btn-primary active float-right disabled\" role=\"button\">View results</a>";
         }
         $x += "</h5>";
         $x += "</div>";
         $x += "<div id=\"collapse" + i + "\" class=\"collapse\" aria-labelledby=\"heading" + i +"\" data-parent=\"#all_panels\">";

         $x += "<ul class=\"list-group list-group-flush\">";

         $x += "<li class=\"list-group-item\">" + "Baseline URL" + ": " + item.baseline_url + "</li>";
         $x += "<li class=\"list-group-item\">" + "Updated URL" + ": " + item.updated_url + "</li>";
         $x += "<li class=\"list-group-item\">" + "Max depth for crawler" + ": " + item.max_depth + "</li>";
         $x += "<li class=\"list-group-item\">" + "Max URLs for crawler" + ": " + item.max_urls + "</li>";
         $x += "<li class=\"list-group-item\">" + "Prefix" + ": " + item.prefix + "</li>";
         if(item.status === "Done"){
            $x += "<li class=\"list-group-item list-group-item-success\">" + "Status" + ": " + item.status + "</li>";

         }
         else if(item.status === "In progress"){
           $x += "<li class=\"list-group-item list-group-item-info\">" + "Status" + ": " + item.status + "</li>";
         }
         else {
            $x += "<li class=\"list-group-item list-group-item-danger\">" + "Status" + ": " + item.status + "</li>";

         }
         $x += "<li class=\"list-group-item\">" + "Started at" + ": " + item.started_at + "</li>";
         $x += "<li class=\"list-group-item\">" + "Stopped at" + ": " + item.stopped_at + "</li>";
         if(item.auth_baseline_username !== "") {
           $x += "<li class=\"list-group-item\">" + "Baseline URL username" + ": " + item.auth_baseline_username + "</li>";
         }
         if(item.auth_baseline_password !== "") {
           $x += "<li class=\"list-group-item\">" + "Baseline URL password" + ": " + item.auth_baseline_password + "</li>";
         }
         if(item.auth_updated_username !== "") {
           $x += "<li class=\"list-group-item\">" + "Updated URL username" + ": " + item.auth_updated_username + "</li>";
         }
         if(item.auth_updated_password !== "") {
           $x += "<li class=\"list-group-item\">" + "Updated URL username" + ": " + item.auth_updated_password + "</li>";
         }


         $x += "</ul>";
         $x += "</div>";
         $x += "</div>";

         $("#all_panels").append($x);

       });
     }
 });
});
