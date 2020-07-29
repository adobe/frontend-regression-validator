function rename_page(page) {
    return page.split('@').join('/');
}

$(function(){
    $.ajax({
        type: "GET",
        url: "/api/viewjobs",
        dataType: 'json', // added data type
        cache: false,
        error: function(response) {console.log("Error");console.log(response);},
        success : function (response) {
            console.log("Success");
            console.log(response);

            // Create items array
            jobs = response["jobs"]
            var items = Object.keys(jobs).map(function(key) {return [key, jobs[key]];});

            // Sort the array based on the second element
            items.sort(function(first, second) {return second[0].substring(0,20).localeCompare(first[0].substring(0,20));});

            for(var iterator in items) {
                job_id = items[iterator][0]
                job = items[iterator][1]
                var input_data = job["input_data"] || {};
                var stats = job["stats"] || {};
                console.log(job)

                status = job["status"]
                error = job["error"] || ""
                baseline_url = input_data["baseline_url"]
                updated_url = input_data["updated_url"]
                max_depth = input_data["max_depth"]
                max_urls = input_data["max_urls"]
                ml_enable = input_data["ml_enable"]
                ml_address = input_data["ml_address"]
                auth_baseline_username = input_data["auth_baseline_username"] || ""
                auth_baseline_password = input_data["auth_baseline_password"] || ""
                auth_updated_username = input_data["auth_updated_username"] || ""
                auth_updated_password = input_data["auth_updated_password"] || ""

                stats_queued_at = stats["queued_at"] || ""
                if (stats_queued_at !== "")
                    stats_queued_at = stats_queued_at.substring(0,19)
                stats_finished_at = stats["finished_at"] || ""
                if (stats_finished_at !== "")
                    stats_finished_at = stats_finished_at.substring(0,19)
                stats_cr_started_at = stats["cr_started_at"] || ""
                if (stats_cr_started_at !== "")
                    stats_cr_started_at = stats_cr_started_at.substring(0,19)
                stats_cr_finished_at = stats["cr_finished_at"] || ""
                if (stats_cr_finished_at !== "")
                    stats_cr_finished_at = stats_cr_finished_at.substring(0,19)
                stats_ml_started_at = stats["ml_started_at"] || ""
                if (stats_ml_started_at !== "")
                    stats_ml_started_at = stats_ml_started_at.substring(0,19)
                stats_ml_finished_at =stats["ml_finished_at"] || ""
                if (stats_ml_finished_at !== "")
                    stats_ml_finished_at = stats_ml_finished_at.substring(0,19)

                console.log(status + " [" + error+"]")
                job_id_with_dots = job_id
                job_id = rename_page(job_id)

                var $x = "";

                $x += "<div class=\"card\">";
                $x += "<div class=\"card-header\" id=\"heading" + job_id + "\">";
                $x += "<h5 class=\"mb-0\">";
                if(status === "Done" && error ===""){
                    $x += "<button class=\"btn btn-success\" type=\"button\" data-toggle=\"collapse\" data-target=\"#collapse"+job_id+"\" aria-expanded=\"true\" aria-controls=\"collapse"+job_id+"\">";

                }
                else if(status === "Done") {
                    $x += "<button class=\"btn btn-danger\" type=\"button\" data-toggle=\"collapse\" data-target=\"#collapse"+job_id+"\" aria-expanded=\"true\" aria-controls=\"collapse"+job_id+"\">";
                }
                else {
                    $x += "<button class=\"btn btn-warning\" type=\"button\" data-toggle=\"collapse\" data-target=\"#collapse"+job_id+"\" aria-expanded=\"true\" aria-controls=\"collapse"+job_id+"\">";
                }
                $x += job_id_with_dots;
                $x += "</button>";
                if(status === "Done") {
                    $x += "<a href=\"/static/results.html?id="+ job_id +" \" class=\"btn btn-primary active float-right\" role=\"button\">View results</a>";
                }
                else {
                    $x += "<a href=\"/static/results.html?id="+ job_id +" \" class=\"btn btn-primary active float-right \" role=\"button\">View results</a>"; // disabled
                }
                $x += "</h5>";
                $x += "</div>";
                $x += "<div id=\"collapse" + job_id.split(".").join("_") + "\" class=\"collapse\" aria-labelledby=\"heading" + job_id.split(".").join("_") +"\" data-parent=\"#all_panels\">";

                $x += "<ul class=\"list-group list-group-flush\">";

                // status
                if(status === "Done" && error === ""){ // done
                    $x += "<li class=\"list-group-item list-group-item-success\">" + "Status" + ":  <b>" + status + "</b></li>";
                }
                else if(status === "Done" && error !== ""){ // error
                    $x += "<li class=\"list-group-item list-group-item-danger\">" + "Status" + ": <b>" + status + "</b> / Error message: <b>" + error + "</b> </li>";
                }
                else { // in progress
                    $x += "<li class=\"list-group-item list-group-item-info\">" + "Status" + ":  <b>" + status + "</b></li>";

                }// to do how to handle partial jobs

                $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>Baseline URL:</span> " + baseline_url + "</li>";
                $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>Updated URL:</span> " + updated_url + "</li>";
                $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>Max URLs/depth for crawler:</span> " + max_urls + " <span style='color:#999999'>/</span> "+ max_depth + "</li>";
                $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>Job Queued / Finished:</span> " + stats_queued_at + " <span style='color:#999999'>/</span> " + stats_finished_at + "</li>";
                $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>Crawling Started / Finished:</span> " + stats_cr_started_at + " <span style='color:#999999'>/</span> " + stats_cr_finished_at + "</li>";
                $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>ML Started / Finished:</span> " + stats_ml_started_at + " <span style='color:#999999'>/</span> " + stats_ml_finished_at + "</li>";

                if (ml_enable===true)
                    $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>ML component URL:</span> " + ml_address + "</li>";


                if(auth_baseline_username !== "") {
                    $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>Baseline URL username:</span> " + auth_baseline_username + "</li>";
                }
                if(auth_baseline_password !== "") {
                    $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>Baseline URL password:</span> " + auth_baseline_password + "</li>";
                }
                if(auth_updated_username !== "") {
                    $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>Updated URL username:</span> " + auth_updated_username + "</li>";
                }
                if(auth_updated_password !== "") {
                    $x += "<li class=\"list-group-item\">" + "<span style='color:#999999'>Updated URL username:</span> " + auth_updated_password + "</li>";
                }


                $x += "</ul>";
                $x += "</div>";
                $x += "</div>";

                $("#all_panels").append($x);

            }; //);
        }
    });
});
