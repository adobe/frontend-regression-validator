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

function rename_page(page) {
    return page.split('@').join('/');
}

var global_response;
var current_id;

$(function(){
    arr = {"id": getQueryVariable("id")}
    $.ajax({
        type: "GET",
        url: "/api/result",
        data: arr,
        success: function(job) {
            global_response = job;
            console.log(job);

            // status and top header and details
            if (job["status"] == "Done" && job["error"] == "") {
                status_css = "bg-success";
            }
            else if (job["status"] == "Done" && job["error"] != "") {
                status_css = "bg-danger";
            }
            else {
                status_css = "bg-warning";
            }
            $("#page-header-top").append('<div class="card text-white '+status_css+'"><div class="card-body">'+job["input_data"]["job_id"]+'</div></div>');
            details = "Status: "+job["status"];
            if (job["error"] !="") { details += " with error: ["+job["error"]+"]"; }
            details += "<br>Job started at "+job["stats"]["queued_at"];
            details += "<br>Job finished at "+job["stats"]["finished_at"];
            $("#page-header-details").append(details);

            // if the job has an error, just display the json
            if (status_css == "bg-danger") {
                $("#page").append("<div class='col-10'>&nbsp;&nbsp;&nbsp;This job has failed with error: &nbsp;<strong>"+job["error"]+"</strong>.<br><br><hr><pre id='json-viewer'></pre><hr></div>")
                $('#json-viewer').jsonViewer(job, {withQuotes:false, withLinks:false, rootCollapsable:false});
                return;
            }
            // if page is still crawling, display json and reload
            if (status_css == "bg-warning") {
                $("#page").append("<div class='col-10'>&nbsp;&nbsp;&nbsp;FRED is&nbsp;<strong>"+job["status"].toLowerCase()+"</strong>, auto-refreshing every 30 seconds ...<br><br><hr><pre id='json-viewer'></pre><hr></div>")
                setTimeout(function() { location.reload(true); }, 30000);
                $('#json-viewer').jsonViewer(job, {withQuotes:false, withLinks:false, rootCollapsable:false});
                return;
            }

            var $p = "";

            // WRITE LEFT MENU
            $p+='<div class="col-2">\n'
            $p+='  <div class="nav flex-column nav-pills" id="v-pills-tab" role="tablist" aria-orientation="vertical">\n';
            $p+='    <a class="nav-link active" id="v-pills-summary-tab" data-toggle="pill" href="#v-pills-summary" role="tab" aria-controls="v-pills-summary" aria-selected="true"><strong>Summary</strong></a>\n';
            var page_id = -1;
            $.each(job["pages"], function( page, dict ) {
                page_id = page_id + 1;
                console.log( "Adding page with id "+page_id+" to menu: "+ page );
                $p+='    <a class="nav-link" id="v-pills-'+page_id+'-tab" data-toggle="pill" href="#v-pills-'+page_id+'" role="tab" aria-controls="v-pills-'+page_id+'" aria-selected="false" style="text-overflow: ellipsis; overflow: hidden; word-wrap: break-word; max-width:100%">'+rename_page(page)+'</a>\n';
                $p+='    <a class="nav-link" id="v-pills-'+page_id+'_network-tab" data-toggle="pill" href="#v-pills-'+page_id+'_network" role="tab" aria-controls="v-pills-'+page_id+'_network" aria-selected="false" style="padding:0;padding-left:2em"><small>Network</small></a>\n';
                $p+='    <a class="nav-link" id="v-pills-'+page_id+'_raw-tab" data-toggle="pill" href="#v-pills-'+page_id+'_raw" role="tab" aria-controls="v-pills-'+page_id+'_raw" aria-selected="false" style="padding:0;padding-left:2em"><small>Visual</small></a>\n';
                if (job["input_data"]["ml_enable"]) {
                    $p+='    <a class="nav-link" id="v-pills-'+page_id+'_ml-tab" data-toggle="pill" href="#v-pills-'+page_id+'_ml" role="tab" aria-controls="v-pills-'+page_id+'_ml" aria-selected="false" style="padding:0;padding-left:2em"><small>Visual (AI)</small></a>\n'; // <span style="color:red;">⁕</span>
                }
            });
            $p+='  </div>\n';
            $p+='</div>\n';

            // WRITE CONTENT
            $p+='<div class="col-10">\n';
            $p+='  <div class="tab-content" id="v-pills-tabContent">\n';

            // SITE SUMMARY PAGE
            $p+='    <!-- start of [SUMMARY] page -->\n';
            $p+='    <div class="tab-pane fade show active" id="v-pills-summary" role="tabpanel" aria-labelledby="v-pills-summary-tab" style="padding-left:0rem;padding-right:0rem;">\n';
            $p+=tabContent_MainSummaryPage(job);
            $p+='    </div>\n';
            $p+='    <!-- end of [SUMMARY] page -->\n';

            // Adding all other pages
            var page_id = -1;
            $.each(job["pages"], function( page, dict ) {
                page_id = page_id + 1;
                console.log( "Processing page with id "+page_id+": "+ page );

                /*
                    MAIN PAGE
                */
                $p+='      <!-- start of ['+page+'] main page -->\n';
                $p+='      <div class="tab-pane fade" id="v-pills-'+page_id+'" role="tabpanel" aria-labelledby="v-pills-'+page_id+'-tab" style="padding-left:0rem;padding-right:0rem;">\n'
                $p+=tabContent_MainPage(page, page_id, dict);
                $p+='      </div>\n';
                $p+='      <!-- end of ['+page+'] main page -->\n';
                /* ****************************************************************************************************** /*

                /*
                    NETWORK PAGE
                */
                $p+='      <!-- start of ['+page+'] network page -->\n';
                $p+='      <div class="tab-pane fade" id="v-pills-'+page_id+'_network" role="tabpanel" aria-labelledby="v-pills-'+page_id+'_network-tab" style="padding-left:0rem;padding-right:0rem;">\n'
                $p+=tabContent_NetworkPage(page, page_id, dict);
                $p+='      </div>\n';
                $p+='      <!-- end of ['+page+'] network page -->\n';
                /* ****************************************************************************************************** /*

                /*
                    RAW SCREENSHOT ANALYSIS
                */
                $p+='      <!-- start of ['+page+'] visual page -->\n';
                $p+='      <div class="tab-pane fade" id="v-pills-'+page_id+'_raw" role="tabpanel" aria-labelledby="v-pills-'+page_id+'_raw-tab" style="padding-left:0rem;padding-right:0rem;">\n'
                $p+=tabContent_Visual(page, page_id, dict);
                $p+='      </div>\n';
                $p+='      <!-- start of ['+page+'] visual page -->\n';
                /* ****************************************************************************************************** /*

                /*
                    ML ANALYSIS
                */
                $p+='      <!-- start of ['+page+'] ml page -->\n';
                $p+='      <div class="tab-pane fade" id="v-pills-'+page_id+'_ml" role="tabpanel" aria-labelledby="v-pills-'+page_id+'_ml-tab" style="padding-left:0rem;padding-right:0rem;">\n'
                $p+=tabContent_VisualAI(page, page_id, dict);
                $p+='      </div>\n';
                $p+='      <!-- end of ['+page+'] ml page -->\n';

            });


            $p+='  </div>\n';
            $p+='</div>\n';
            $("#page").append($p);

            // summary page init knobs
            overall_divergence = job["report"]["overall_divergence"]
            network_divergence = job["report"]["crawl_divergence"]
            visual_divergence= job["report"]["raw_screenshot_divergence"]
            visual_ai_divergence = job["report"]["ml_screenshot_divergence"]

            initKnob(overall_divergence, 'summary-overall');
            initKnob(network_divergence, 'summary-network');
            initKnob(visual_divergence, 'summary-visual');
            initKnob(visual_ai_divergence, 'summary-visualai');

            $('#json-viewer').jsonViewer(job, {withQuotes:false, withLinks:false, rootCollapsable:false});
            $('[data-toggle="tooltip"]').tooltip()

            $( document ).ready(function() {
                console.log( "DOCUMENT LOADED MARK" );

            });
            console.log("FINISHED JS PAGE")
        },
        error: function(response) {
            $("#wrong_id_alert").toggleClass("d-none");
        }
    });

});

function tabContent_MainSummaryPage(job) {
    var $p = "";
    $p+='        <div class="card border-primary mb-3"><div class="card-body text-primary">Summary for <strong>'+job["input_data"]["job_id"]+'</strong></div></div>\n';

    $p+='    <div class="card-deck">\n';
    $p+='      <div class="card text-center"><div id="summary-overall" style="margin-top:1rem;"></div><div class="card-body"><p class="card-text"><strong>Overall divergence</strong></p></div></div>\n';
    $p+='      <div class="card text-center"><div id="summary-network" style="margin-top:1rem;"></div><div class="card-body"><p class="card-text">Network divergence<br><small style="color:#999999">(weight: '+(parseFloat(job["input_data"]["network_divergence_weight"])*100).toFixed(0)+'%)</small></p></div></div>\n';
    $p+='      <div class="card text-center"><div id="summary-visual" style="margin-top:1rem;"></div><div class="card-body"><p class="card-text">Visual divergence<br><small style="color:#999999">(weight: '+(parseFloat(job["input_data"]["visual_divergence_weight"])*100).toFixed(0)+'%)</small></p></div></div>\n';
    if (job["input_data"]["ml_enable"] == true)
        $p+='      <div class="card text-center"><div id="summary-visualai" style="margin-top:1rem;"></div><div class="card-body"><p class="card-text">Visual divergence (A.I.)<br><small style="color:#999999">(weight: '+(parseFloat(job["input_data"]["visual_divergence_ai_weight"])*100).toFixed(0)+'%)</small></p></div></div>\n';
    else
        $p+='      <div class="card text-center"><div id="summary-visualai" style="margin-top:1rem;"></div><div class="card-body"><p class="card-text">Visual divergence (A.I.)<br><small style="color:#999999">(disabled)</small></p></div></div>\n';
    $p+='     </div>\n';

    // Top divergent pages table
    lst = getMostDifferentVisuallyPages(job["pages"])
    $p+='  <br>\n';
    $p+='  <table class="table table-hover text-xsmall table-striped table-sm" style="font-size:0.825rem"><thead><tr>';
    $p+="    <th class='w-75'>Top visually different pages</th>";
    $p+="    <th class='w-15'>Value</th>";
    $p+="    <th class='w-5'>Resolution</th>";
    $p+="</tr></thead><tbody>";
    for (var i = 0; i < lst["visual"].length; i++) {
        page = lst["visual"][i][0].split("|")[1];
        res = lst["visual"][i][0].split("|")[0];
        val = lst["visual"][i][1];
        if (val>0) {
            $p+='<tr><td>'+rename_page(page)+'</td><td>'+val.toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">'+res+'px</td></tr>';
        }
    }
    $p+="</tbody></table>\n";

    // Top divergent AI pages table
    if (job["input_data"]["ml_enable"] == true) {
        $p+='  <br>\n';
        $p+='  <table class="table table-hover text-xsmall table-striped table-sm" style="font-size:0.825rem"><thead><tr>';
        $p+="    <th class='w-75'>Top visually different pages (A.I. analyzed) </th>";
        $p+="    <th class='w-15'>Value</th>";
        $p+="    <th class='w-5'>Resolution</th>";
        $p+="</tr></thead><tbody>";
        for (var i = 0; i < lst["visual_ai"].length; i++) {
            page = lst["visual_ai"][i][0].split("|")[1];
            res = lst["visual_ai"][i][0].split("|")[0];
            val = lst["visual_ai"][i][1];
            if (val>0) {
                $p+='<tr><td>'+rename_page(page)+'</td><td>'+val.toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">'+res+'px</td></tr>';
            }
        }
        $p+="</tbody></table>\n";
    }
    // Summary table
    $p+='  <br><table class="table table-hover text-xsmall table-striped table-sm" style="font-size:0.825rem"><thead><tr>';
    $p+="    <th>Site summary</th>";
    $p+="    <th>Value</th>";
    $p+="    <th>Help</th>";
    $p+="</tr></thead><tbody>";
    $p+='<tr><td>Network divergence:</td><td>'+job["report"]["crawl_divergence"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
    $p+='<tr><td>Visual divergence - Overall:</td><td>'+job["report"]["raw_screenshot_divergence"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 100% = identical</td></tr>';
    $p+='<tr><td>&nbsp;&nbsp;Visual divergence - Normalized Square Error divergence (NMSE):</td><td style="color:#606060;">'+job["report"]["raw_screenshot_mse"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 100% = identical</td></tr>';
    $p+='<tr><td>&nbsp;&nbsp;Visual divergence - Structural Similarity Index (SSIM):</td><td style="color:#606060;">'+job["report"]["raw_screenshot_ssim"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 100% = identical</td></tr>';
    $p+='<tr><td>&nbsp;&nbsp;Visual divergence - Percent different pixels:</td><td style="color:#606060;">'+job["report"]["raw_screenshot_diff_pixels"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 100% = identical</td></tr>';
    if (job["input_data"]["ml_enable"]) { // ML divergence, if it exists
        $p+='<tr><td>Visual divergence (A.I.) - Overall:</td><td>'+job["report"]["ml_screenshot_divergence"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 100% = identical</td></tr>';
    }
    $p+="</tbody></table>\n";
    $p+='  <br><h6>Full result JSON:</h6>\n';
    $p+='  <pre id="json-viewer"></pre>\n';
    return $p;
}

function tabContent_MainPage(page, page_id, dict) {
    var $p="";
    $p+='        <div class="card border-primary mb-3"><div class="card-body text-primary"><strong>'+rename_page(page)+'</strong>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp Summary</div></div>';

    $p+='<br><table class="table table-hover text-xsmall table-striped table-sm" style="font-size:0.825rem">\n<thead><tr>';
    $p+="    <th>&nbsp;</th>";
    $p+="    <th>Value</th>";
    $p+="    <th>Help</th>";
    $p+="</tr></thead><tbody>";
    $p+='<tr><td>Console log divergence</td><td>'+dict["console_logs"]["divergence"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0 = identical</td></tr>';
    $p+='<tr><td>Network log divergence</td><td>'+dict["network_logs"]["divergence"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0 = identical</td></tr>';
    $.each(dict["raw_screenshot_analysis"], function( res, obj ) {
        $p+='<tr><td>Visual divergence at '+res+'px - Normalized Square Error divergence (NMSE):</td><td>'+obj["mse"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>Visual divergence at '+res+'px - Structural Similarity Index (SSIM): </td><td>'+obj["ssim"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 100% = identical</td></tr>';
        $p+='<tr><td>Visual divergence at '+res+'px - Percent different pixels:</td><td>'+obj["diff_pixels"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
    });

    $.each(dict["ml_screenshot_analysis"], function( res, obj ) { // for each resolution, write tab-content
        if (isNaN(parseInt(res))) {return;}
        $p+='<tr><td>Overall visual divergence (A.I.) at '+res+'px:</td><td>'+obj["overall"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>&nbsp;&nbsp;&nbsp;Mask overall divergence at '+res+'px:</td><td>'+obj["mask"]["overall"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>&nbsp;&nbsp;&nbsp;Content overall divergence at '+res+'px:</td><td>'+obj["content"]["overall"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';

    });

    $p+="</tbody></table>\n";
    return $p;
}

function tabContent_NetworkPage(page, page_id, dict) {
    var $p="";
    $p+='        <div class="card border-primary mb-3"><div class="card-body text-primary"><strong>'+rename_page(page)+'</strong>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp Network Logs</div></div>';
    $p+='<h6>Network logs divergence: '+dict["network_logs"]["divergence"].toFixed(2)+' %</h6>\n';

    if (dict["network_logs"]["report"]["on_baseline_not_on_updated"].length > 0) {
        $p+='Items that are in the baseline page and not in the updated page:\n';
        $p+='<br><table class="table table-borderless table-hover text-xsmall table-striped table-sm" style="font-size:0.625em;">\n';
        $.each(dict["network_logs"]["report"]["on_baseline_not_on_updated"], function( i, line ) {
            $p+='<tr><td>'+line+'</td></tr>\n';
        });
        $p+='</tbody></table>\n';
    }
    if (dict["network_logs"]["report"]["on_updated_not_on_baseline"].length > 0) {
        $p+='Items that are in the updated page and not in the baseline page:\n';
        $p+='<br><table class="table table-borderless table-hover text-xsmall table-striped table-sm" style="font-size:0.625em;">\n';
        $.each(dict["network_logs"]["report"]["on_updated_not_on_baseline"], function( i, line ) {
            $p+='<tr><td>'+line+'</td></tr>\n';
        });
        $p+='</tbody></table>\n';
    }

    $p+='<br><h6>Console logs divergence: '+dict["console_logs"]["divergence"].toFixed(2)+' %</h6>\n';

    if (dict["console_logs"]["report"]["on_baseline_not_on_updated"].length > 0) {
        $p+='Items that are in the baseline page and not in the updated page:\n';
        $p+='<br><table class="table table-borderless table-hover text-xsmall table-striped table-sm" style="font-size:0.625em;">\n';
        $.each(dict["console_logs"]["report"]["on_baseline_not_on_updated"], function( i, line ) {
            $p+='<tr><td>'+line+'</td></tr>\n';
        });
        $p+='</tbody></table>\n';
    }
    if (dict["console_logs"]["report"]["on_updated_not_on_baseline"].length > 0) {
        $p+='Items that are in the updated page and not in the baseline page:\n';
        $p+='<br><table class="table table-borderless table-hover text-xsmall table-striped table-sm" style="font-size:0.625em;">\n';
        $.each(dict["console_logs"]["report"]["on_updated_not_on_baseline"], function( i, line ) {
            $p+='<tr><td>'+line+'</td></tr>\n';
        });
        $p+='</tbody></table>\n';
    }
    return $p;
}

function tabContent_Visual(page, page_id, dict) {
    var $p="";
    $p+='        <div class="card border-primary mb-3"><div class="card-body text-primary"><strong>'+rename_page(page)+'</strong>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp Visual Analysis</div></div>\n';

    // write horizontal tab header with resolutions
    $p+='          <ul class="nav nav-pills mb-3" id="pills-tab-'+page_id+'" role="tablist">\n';
    var active = " active"
    $.each(dict["raw_screenshot_analysis"], function( res, obj ) {
        $p+='            <li class="nav-item"><a class="nav-link'+active+'" id="pills-'+res+'-tab-'+page_id+'" data-toggle="pill" href="#pills-'+res+'-'+page_id+'" role="tab" aria-controls="pills-'+res+'-'+page_id+'" aria-selected="false">'+res+'px</a></li>\n';
        active = "";
    });
    $p+='          </ul>\n';

    // fill each tab with content
    var active = " active show";
    $p+='          <div class="tab-content justify-content-center mx-auto" id="pills-tabContent-'+page_id+'">\n';
    $.each(dict["raw_screenshot_analysis"], function( res, obj ) { // for each resolution, write tab-content
        $p+='            <div class="tab-pane fade'+active+'" id="pills-'+res+'-'+page_id+'" role="tabpanel" aria-labelledby="pills-'+res+'-tab-'+page_id+'">';
        active = "";

        $p+='<table class="table table-hover text-xsmall table-striped table-sm" style="font-size:0.825rem"><tbody>\n';
        $p+='<tr><td>Normalized Square Error divergence (NMSE):</td><td>'+obj["mse"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>Structural Similarity Index (SSIM): </td><td>'+obj["ssim"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 100% = identical</td></tr>';
        $p+='<tr><td>Percent different pixels:</td><td>'+obj["diff_pixels"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+="</tbody></table>\n";

        // write carousel
        $p+='            <div class="d-block w-100 justify-content-center mx-auto" style="text-align: center; margin: auto;">\n';
        $p+='                <div style="color:#999999;font-size:0.8em" data-toggle="tooltip" data-html="true" title="<strong>BASELINE:</strong> The baseline screenshot with differences highlighted in red<br><strong>UPDATED:</strong> The updated screenshot with differences highlighted in red<br><strong>DIFFERENCE:</strong> The updated screenshot in grayscale with the different regions filled with red<br><strong>THRESHOLD:</strong> The updated screenshot in black with changes in white for fast location identification">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>(Help: Navigate by clicking on the sides of the image)</em></div>\n';
        $p+='                <div id="carousel-'+res+'-'+page_id+'" class="d-block carousel border border-primary" data-ride="carousel" data-interval="false" style="margin: auto;">\n';
        $p+='                  <div class="carousel-inner">\n';
        $p+='                    <div class="carousel-item active"><div class="w-100 font-weight-bold lead small" style="text-align:center;">BASELINE</div>\n';
        $p+='                      <img class="d-block w-100" src="'+obj["baseline_file_path_highlight"]+'" alt="BASELINE">\n';
        $p+='                    </div>\n';
        $p+='                    <div class="carousel-item"><div class="w-100 font-weight-bold lead small" style="text-align:center;">UPDATED</div>\n';
        $p+='                      <img class="d-block w-100" src="'+obj["updated_file_path_highlight"]+'" alt="UPDATED">\n';
        $p+='                    </div>\n';
        $p+='                    <div class="carousel-item"><div class="w-100 font-weight-bold lead small" style="text-align:center;">DIFFERENCE</div>\n';
        $p+='                      <img class="d-block w-100" src="'+obj["updated_file_path_difference"]+'" alt="DIFFERENCE">\n';
        $p+='                    </div>\n';
        $p+='                    <div class="carousel-item"><div class="w-100 font-weight-bold lead small" style="text-align:center;">THRESHOLD</div>\n';
        $p+='                      <img class="d-block w-100" src="'+obj["updated_file_path_threshold"]+'" alt="THRESHOLD">\n';
        $p+='                    </div>\n';
        $p+='                  </div>\n';
        $p+='                  <a class="carousel-control-prev" href="#carousel-'+res+'-'+page_id+'" role="button" data-slide="prev">\n';
        //$p+='                    <span class="carousel-control-prev-icon" aria-hidden="true"></span><span class="sr-only">Previous</span>\n';
        $p+='                  </a>\n';
        $p+='                  <a class="carousel-control-next" href="#carousel-'+res+'-'+page_id+'" role="button" data-slide="next">\n';
        //$p+='                    <span class="carousel-control-next-icon" aria-hidden="true"></span><span class="sr-only">Next</span>\n';
        $p+='                  </a>\n';
        $p+='                </div>\n';
        $p+='              </div>\n';

        $p+='            </div>\n'; // tab pane
    });
    $p+='          </div>\n'; // tab-content
    return $p;
}

function tabContent_VisualAI(page, page_id, dict) {
    var $p = ""
    $p+='        <div class="card border-primary mb-3"><div class="card-body text-primary"><strong>'+rename_page(page)+'</strong>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp Visual Analysis (A.I)</div></div>';

    // write horizontal tab header with resolutions
    $p+='          <ul class="nav nav-pills mb-3" id="pills-tab-ai-'+page_id+'" role="tablist">\n';
    var active = " active"
    $.each(dict["ml_screenshot_analysis"], function( res, obj ) {
        if (isNaN(parseInt(res))) {return;}
        $p+='            <li class="nav-item"><a class="nav-link'+active+'" id="pills-ai-'+res+'-tab-'+page_id+'" data-toggle="pill" href="#pills-ai-'+res+'-'+page_id+'" role="tab" aria-controls="pills-ai-'+res+'-'+page_id+'" aria-selected="false">'+res+'px</a></li>\n';
        active = "";
    });
    $p+='          </ul>\n';

    // fill each tab with content
    var active = " active show";
    $p+='          <div class="tab-content justify-content-center mx-auto" id="pills-tabContent-ai-'+page_id+'">\n';
    $.each(dict["ml_screenshot_analysis"], function( res, obj ) { // for each resolution, write tab-content
        if (isNaN(parseInt(res))) {return;}
        $p+='            <div class="tab-pane fade'+active+'" id="pills-ai-'+res+'-'+page_id+'" role="tabpanel" aria-labelledby="pills-ai-'+res+'-tab-'+page_id+'">';
        active = "";
        //console.log(obj)
        $p+='<table class="table table-hover text-xsmall table-striped table-sm" style="font-size:0.825rem"><tbody>\n';
        $p+='<tr><td>Overall divergence:</td><td>'+obj["overall"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>Mask overall divergence: </td><td>'+obj["mask"]["overall"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>&nbsp;&nbsp;&nbsp;Mask textblocks divergence: </td><td>'+obj["mask"]["textblock"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>&nbsp;&nbsp;&nbsp;Mask images divergence: </td><td>'+obj["mask"]["images"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>Content overall divergence:</td><td>'+obj["content"]["overall"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>&nbsp;&nbsp;&nbsp;Content textblocks divergence: </td><td>'+obj["content"]["textblock"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';
        $p+='<tr><td>&nbsp;&nbsp;&nbsp;Content images divergence: </td><td>'+obj["content"]["images"].toFixed(2)+' %</td><td style="color:#606060;font-size:0.8em">[0,100] where 0% = identical</td></tr>';

        $p+="</tbody></table>\n";


        $p+='<div class="modal-body row">\n';
        $p+='<div class="col-md-6">\n';
        // write textblock carousel
        $p+='            <div class="d-block w-100 justify-content-center mx-auto" style="text-align: center; margin: auto;">\n';
        $p+='                <div style="color:#999999;font-size:0.8em" data-toggle="tooltip" data-html="true" title="<strong>BASELINE:</strong> The baseline screenshot with differences highlighted in red<br><strong>UPDATED:</strong> The updated screenshot with differences highlighted in red<br><strong>DIFFERENCE:</strong> The updated screenshot in grayscale with the different regions filled with red<br><strong>THRESHOLD:</strong> The updated screenshot in black with changes in white for fast location identification">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>(Help: Navigate by clicking on the sides of the image)</em></div>\n';
        $p+='                <div id="carousel-ai-textblock-'+res+'-'+page_id+'" class="d-block carousel border border-primary" data-ride="carousel" data-interval="false" style="margin: auto;">\n';
        $p+='                  <div class="carousel-inner">\n';
        $p+='                    <div class="carousel-item active"><div class="w-100 font-weight-bold lead small" style="text-align:center;">TEXTBLOCKS: BASELINE</div>\n';
        $p+='                      <img class="d-block w-100" src="'+obj["textblock_baseline_file_path"]+'" alt="TEXTBLOCKS: BASELINE">\n';
        $p+='                    </div>\n';
        $p+='                    <div class="carousel-item"><div class="w-100 font-weight-bold lead small" style="text-align:center;">TEXTBLOCKS: UPDATED</div>\n';
        $p+='                      <img class="d-block w-100" src="'+obj["textblock_updated_file_path"]+'" alt="TEXTBLOCKS: UPDATED">\n';
        $p+='                    </div>\n';
        $p+='                  </div>\n';
        $p+='                  <a class="carousel-control-prev" href="#carousel-ai-textblock-'+res+'-'+page_id+'" role="button" data-slide="prev">\n';
        $p+='                  </a>\n';
        $p+='                  <a class="carousel-control-next" href="#carousel-ai-textblock-'+res+'-'+page_id+'" role="button" data-slide="next">\n';
        $p+='                  </a>\n';
        $p+='                </div>\n';
        $p+='              </div>\n';


        $p+='</div>\n'; // end column 1/2
        $p+='<div class="col-md-6">\n'; // start column 2/2

        // write images carousel
        $p+='            <div class="d-block w-100 justify-content-center mx-auto" style="text-align: center; margin: auto;">\n';
        $p+='                <div style="color:#999999;font-size:0.8em" data-toggle="tooltip" data-html="true" title="<strong>BASELINE:</strong> The baseline screenshot with differences highlighted in red<br><strong>UPDATED:</strong> The updated screenshot with differences highlighted in red<br><strong>DIFFERENCE:</strong> The updated screenshot in grayscale with the different regions filled with red<br><strong>THRESHOLD:</strong> The updated screenshot in black with changes in white for fast location identification">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>(Help: Navigate by clicking on the sides of the image)</em></div>\n';
        $p+='                <div id="carousel-ai-images-'+res+'-'+page_id+'" class="d-block carousel border border-primary" data-ride="carousel" data-interval="false" style="margin: auto;">\n';
        $p+='                  <div class="carousel-inner">\n';
        $p+='                    <div class="carousel-item active"><div class="w-100 font-weight-bold lead small" style="text-align:center;">IMAGES: BASELINE</div>\n';
        $p+='                      <img class="d-block w-100" src="'+obj["images_baseline_file_path"]+'" alt="IMAGES: BASELINE">\n';
        $p+='                    </div>\n';
        $p+='                    <div class="carousel-item"><div class="w-100 font-weight-bold lead small" style="text-align:center;">IMAGES: UPDATED</div>\n';
        $p+='                      <img class="d-block w-100" src="'+obj["images_updated_file_path"]+'" alt="IMAGES: UPDATED">\n';
        $p+='                    </div>\n';
        $p+='                  </div>\n';
        $p+='                  <a class="carousel-control-prev" href="#carousel-ai-images-'+res+'-'+page_id+'" role="button" data-slide="prev">\n';
        $p+='                  </a>\n';
        $p+='                  <a class="carousel-control-next" href="#carousel-ai-images-'+res+'-'+page_id+'" role="button" data-slide="next">\n';
        $p+='                  </a>\n';
        $p+='                </div>\n';
        $p+='              </div>\n';

        $p+='</div>\n'; // end column 2/2
        $p+='</div>\n'; // end two columns

        $p+='            </div>\n'; // tab pane
    });
    $p+='          </div>\n'; // tab-content
    //$p+=JSON.stringify(dict)
    return $p;
}

function getMostDifferentVisuallyPages(pages) {
    var top_visual = {}
    var top_visual_ai = {}
    $.each(pages, function( page, dict ) {
        $.each(dict["raw_screenshot_analysis"], function(res, obj) {
            key = res+"|"+page
            val = (obj["diff_pixels"]+(100-obj["ssim"])+obj["mse"])/3
            top_visual[key] = val
        });
        $.each(dict["ml_screenshot_analysis"], function(res, obj) {
            key = res+"|"+page
            val = obj["overall"]
            top_visual_ai[key] = val
        });
    });

    result = {}

    // Create items array
    var items = Object.keys(top_visual).map(function(key) {return [key, top_visual[key]];});
    // Sort the array based on the second element
    items.sort(function(first, second) {return second[1] - first[1];});
    // Create a new array with only the first items
    result["visual"] = items.slice(0, 6);

    // Create items array
    var items = Object.keys(top_visual_ai).map(function(key) {return [key, top_visual_ai[key]];});
    // Sort the array based on the second element
    items.sort(function(first, second) {return second[1] - first[1];});
    // Create a new array with only the first items
    result["visual_ai"] = items.slice(0, 6);

    return result
}

function initKnob(value, id) {
    var knob = pureknob.createKnob(150, 150);
    knob.setValue(value);
    knob.setProperty('angleStart', -0.75 * Math.PI);
    knob.setProperty('angleEnd', 0.75 * Math.PI);
    knob.setProperty('colorFG', '#FF3333');
    knob.setProperty('colorBG', '#DDDDDD');
    knob.setProperty('trackWidth', 0.4);
    knob.setProperty('valMin', 0);
    knob.setProperty('valMax', 100);
    knob.setProperty('readonly', true);
    var elem = document.getElementById(id);
    elem.appendChild(knob.node());
}
