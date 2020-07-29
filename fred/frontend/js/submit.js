$(document).ready(function(){

    $("#require_auth").prop('checked', false);
});
$("button").click(function(e) {
    e.preventDefault();
    arr = {
        "baseline_url": $("#baseline_url").val(),
        "updated_url": $("#updated_url").val(),
        "max_depth": $("#max_depth").val(),
        "max_urls": $("#max_urls").val(),
        "prefix": $("#prefix").val(),
        "ml_enable": $("#ml_enable").is(":checked"),
        "ml_address": $("#ml_address").val(),
        "resolutions": $("#resolutions").val(),
        "score_weights": $("#score_weights").val(),
        "score_epsilon": $("#score_epsilon").val()
    };
    console.log(arr);
    $.ajax({
        type: "POST",
        url: "/api/verify",
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify(arr),
        success : function (response) {
            console.log("SUCCESS")
            $("#new_job_alert").text("Submitted new job with ID: " + response.id);
            $("#new_job_alert").removeClass("d-none").fadeIn(100);
            $("#new_job_alert").fadeOut(300).fadeIn(300).fadeOut(300).fadeIn(300);
        },
        error: function(response) {
            console.log("ERROR")
            console.log(response)
            alert("ERROR: "+response.responseJSON["error"]);
        }
    });
});

$("#require_auth").change(function() {
    $("#auth_boxes").toggleClass("d-none");
});

$("#ml_enable").change(function() {
    $("#ml_address_boxes").toggleClass("d-none");
});
