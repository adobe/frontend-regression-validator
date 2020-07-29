$(document).ready(function(){
 //my code here
 $("#require_auth").prop('checked', false);
});
$("button").click(function(e) {
    e.preventDefault();
    arr = {
        "baseline_url": $("#baseline_url").val(),
        "updated_url": $("#updated_url").val(),
        "max_depth": $("#max_depth").val(),
        "max_urls": $("#max_urls").val(),
        "prefix": $("#prefix").val()
    };
    console.log(arr);
    console.log("test");
    $.ajax({
        type: "POST",
        url: "/api/verify",
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify(arr),
        success : function (response) {
          $("#new_job_alert").text("Submitted new job with ID: " + response.id);
          $("#new_job_alert").toggleClass("d-none");
        },
        error: function(response) {
         alert(response.responseJSON.Error);
       }
    });
});
$("#require_auth").change(function() {
    $("#auth_boxes").toggleClass("d-none");
});
