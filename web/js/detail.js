param = new URL(window.location.href).searchParams;
taxi_id = param.get('taxi')
// TODO : AWS s3 주소 입력 필요 
s3_ref = "https://"

init()

function running(){
    if($('#run_btn').val()=='on'){
        $('#run_btn').val("off")
        $('#run_btn').find('img').attr("src","img/interval.png")
        clearInterval(interval);
    }else{
        $('#run_btn').val("on")
        $('#run_btn').find('img').attr("src","img/interval.svg")
        interval = setInterval(init, 2000)
        
    }
}

function init(){
    v = Math.random()*1000
    level = 0
    $('#sensor tr:not(:first)').remove()
    $('#motion tr:not(:first)').remove()
    $('#video tr:not(:first)').remove()
    
    
    $('#detail_img').attr("src",s3_ref+taxi_id+"/preview.jpg?v="+v)
    $('#taxi_name').text(taxi_id)

    $.ajax({
        // TODO : AWS REST api 입력 필요 
        url: "https://",
        data: { type : "single", taxi_id : taxi_id },
        method: "GET",
        dataType: "json",
        success : function(data){
            $.ajax({
                // TODO : kakao api 입력 필요 
                url: "https://?x="+data[0].long+"&y="+data[0].lat,
                // TODO : kakao api 키 입력 필요 
                headers : { "Authorization" : "KakaoAK {api key}"},
                method: "GET",
                dataType: "json",
                async: false,
                success : function(value){
                    $('#loc_cur').text(value.documents[1].address_name+" ("+data[0].lat+", "+data[0].long+")")
                }
            })
        }
    })

    $.ajax({
        // TODO : AWS REST api 입력 필요 
        url: "https://",
        data: { type : "all", taxi_id : taxi_id },
        method: "GET",
        dataType: "json",
        async: false,
        success : function(data){
            $.each(data, function(index, value){
                
                $('#sensor').append("<tr><td>"+value.time+"</td><td>"+value.alcohol+"</td><td>"+value.sound+"</td></tr>")
                if(index==data.length-1){
                    if(value.alcohol=="1"){level+=1}
                    if(value.sound=="1"){level+=1}
                    $('#alcohol_cur').text(value.alcohol)
                    $('#sound_cur').text(value.sound)
                }
            })
        }
    })

    $.ajax({
        // TODO : AWS REST api 입력 필요 
        url: "https://",
        data: { type : "all", taxi_id : taxi_id },
        method: "GET",
        dataType: "json",
        async:false,
        success : function(data){
            $.each(data, function(index, value){
                $('#motion').append("<tr><td>"+value.time+"</td><td>"+value.security+"</td></tr>")
                if(index==data.length-1){
                    $('#motion_cur').text(value.security)
                    if(value.security=="WARNING"){level+=1}
                    if(value.security=="DANGER"){level+=2}
                }
            })
        }
    })

    $.ajax({
        // TODO : AWS REST api 입력 필요 
        url: "https://",
        data: { type : "all", taxi_id : taxi_id },
        method: "GET",
        dataType: "json",
        async:false,
        success : function(data){
            $.each(data, function(index, value){
                $('#video').append("<tr><td><a href='"+s3_ref+value+"'>"+value+"</a></td></tr>")
            })
        }
    })
    if(level>2){$('#secure_level').text("위험")}
    else if(level>1){$('#secure_level').text("주의")}
    else {$('#secure_level').text("양호")}
}