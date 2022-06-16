// TODO : AWS s3 주소 입력 필요 
var s3_ref = "https://"
var map_data = [];
init()

function running(){
    if($('#run_btn').val()=='on'){
        $('#run_btn').val("off")
        $('#run_btn').find('img').attr("src","img/interval.png")
        clearInterval(interval);
    }else{
        $('#run_btn').val("on")
        $('#run_btn').find('img').attr("src","img/interval.svg")
        interval = setInterval(init, 5000)
        
    }
}


function init(){
    v = Math.random()*1000
    $('#taxi_status').empty()
    map_data = [];
    var taxies;

    cnt_all=0;
    cnt_s=0;
    cnt_w=0;
    cnt_d=0;

    $.ajax({
        // TODO : AWS REST api 입력 필요 (location)
        url: "https://",
        data: { type : "all" },
        method: "GET",
        dataType: "json",
        async:false,
        success : function(data){
            taxies = data
            cnt_all = data.length
        }
    })

    $.each(taxies,function(index, value){
        var level = 0; // 2 이상 [주의] 3 이상 [위험]
        var html1 = ""
        var html2 = '<div class="col-lg-10 col-lg-offset-1 mini_status">'

        $.ajax({
            // TODO : AWS REST api 입력 필요 (sensor)
            url: "https://",
            data: { type : "single", taxi_id : value.taxi_id },
            method: "GET",
            dataType: "json",
            async: false,
            success : function(data){
                if(data.length<1){
                    html2 += '<div class="col-lg-4"><h4>음주</h4><img src="img/green_led.png"></div>'
                    html2 += '<div class="col-lg-4"><h4>폭언</h4><img src="img/green_led.png"></div>'
                }
                else{
                    if(data[0].alcohol == '1'){
                        level+=1
                        html2 += '<div class="col-lg-4"><h4>음주</h4><img src="img/red_led.png"></div>'
                    }else {
                        html2 += '<div class="col-lg-4"><h4>음주</h4><img src="img/green_led.png"></div>'
                    }
                    if (data[0].sound == '1'){
                        level+=1
                        html2 += '<div class="col-lg-4"><h4>폭언</h4><img src="img/red_led.png"></div>'
                    }else{
                        html2 += '<div class="col-lg-4"><h4>폭언</h4><img src="img/green_led.png"></div>'
                    }           
                }
            }
        })

        $.ajax({
            // TODO : AWS REST api 입력 필요 (motion)
            url: "https://",
            data: { type : "single", taxi_id : value.taxi_id },
            method: "GET",
            dataType: "json",
            async:false,
            success : function(data){
                if(data<1){
                    html2 += '<div class="col-lg-4"><h4>폭력</h4><img src="img/green_led.png"></div>'
                }else{
                    if(data[0].security == "WARNING") {
                        level +=1
                        html2 += '<div class="col-lg-4"><h4>폭력</h4><img src="img/yellow_led.png"></div>'
                    }else if(data[0].security == "DANGER"){
                        level += 3
                        html2 += '<div class="col-lg-4"><h4>폭력</h4><img src="img/red_led.png"></div>'
                    }else{
                        html2 += '<div class="col-lg-4"><h4>폭력</h4><img src="img/green_led.png"></div>'
                    }
                }
            }
        })
        
        if (level>2){
            html1 = '<div class="col-lg-4 centered status danger"  onclick="mv_detail(\''+value.taxi_id+'\')">'
            html1 += '<img src="'+s3_ref+value.taxi_id+'/preview.jpg?v='+v+'" onerror=this.src="img/default_img.png">'
            html1 += '<h2><b>(위험)</b>'+value.taxi_id+'</h2>'
            cnt_d += 1
        }else if(level>1){
            html1 = '<div class="col-lg-4 centered status warning"  onclick="mv_detail(\''+value.taxi_id+'\')">'
            html1 += '<img src="'+s3_ref+value.taxi_id+'/preview.jpg?v='+v+'" onerror=this.src="img/default_img.png">'
            html1 += '<h2><b>(주의)</b>'+value.taxi_id+'</h2>'
            cnt_w += 1
        }else{
            html1 = '<div class="col-lg-4 centered status safety"  onclick="mv_detail(\''+value.taxi_id+'\')">'
            html1 += '<img src="'+s3_ref+value.taxi_id+'/preview.jpg?v='+v+'" onerror=this.src="img/default_img.png">'
            html1 += '<h2><b></b>'+value.taxi_id+'</h2>'
            cnt_s += 1
        }
        html1 += html2
        html1 += '</div></div>'

        $('#taxi_status').append(html1)
        map_data.push([value.taxi_id, parseFloat(value.lat), parseFloat(value.long)])
    })
    $('#count_all').text(cnt_all+"")
    $('#count_s').text(cnt_s+"")
    $('#count_w').text(cnt_w+"")
    $('#count_d').text(cnt_d+"")
    myMap()
}

function mv_detail(taxi_id){
    location.href = "detail.html?taxi="+taxi_id
}

function myMap(){
    var mapOptions = { 
        center:new google.maps.LatLng(37.6228,127.0533),
        zoom:15
    };

    var map = new google.maps.Map( 
        document.getElementById("googleMap") 
        , mapOptions );


    $.each(map_data, function(index, value){
        var mainMarker = new google.maps.Marker({
            position: { lat : value[1], lng : value[2]},
            map: map,
            label: {
                text: value[0],
                color: "#ff0000",
                fontWeight: "bold"
            },
            icon: {
                url: "img/mini_taxi.png",
                ScaledSize: new google.maps.Size(40, 40),
                labelOrigin: new google.maps.Point(20, -10)
            }
        });
    })
}

