% This is a wrapper for whos to work better with GPUmat variables.
Awhos = whos;
maxwhosnamel = ones(5,1);
whosize = cell(1,length(Awhos));
whobytes = cell(1,length(Awhos));
whostring = cell(1,length(Awhos));
for iwho = 1:length(Awhos)
    if(strcmp(Awhos(iwho).class,'GPUsingle'))
        whosize{iwho} = eval(['size(' Awhos(iwho).name ');']);
        whobytes{iwho} = prod(whosize{iwho})*4;
    elseif(strcmp(Awhos(iwho).class,'GPUdouble'))
        whosize{iwho} = eval(['size(' Awhos(iwho).name ');']);
        whobytes{iwho} = prod(whosize{iwho})*8;
    else % CPU
        whosize{iwho} = Awhos(iwho).size;
        whobytes{iwho} = Awhos(iwho).bytes;
    end
    whostring{iwho} = '';
    for iwhol=1:length(whosize{iwho})
        whostring{iwho} = [whostring{iwho} sprintf('%dx',whosize{iwho}(iwhol))];
    end
    whostring{iwho} = whostring{iwho}(1:end-1);
    
    % Longest name.
    if(length(Awhos(iwho).name)>maxwhosnamel(1))
        maxwhosnamel(1) = length(Awhos(iwho).name);
    end
    % Longest size string.
    if(length(whostring{iwho})>maxwhosnamel(2))
        maxwhosnamel(2) = length(whostring{iwho});
    end
    % Longest byte string.
    if(length(num2str(whobytes{iwho}))>maxwhosnamel(3))
        maxwhosnamel(3) = length(num2str(whobytes{iwho}));
    end
    % Distance to first x.
    if(regexp(whostring{iwho},'x','once')>maxwhosnamel(4))
        maxwhosnamel(4) = regexp(whostring{iwho},'x','once');
    end
    % Distance from last x to end.
    if((length(whostring{iwho})-regexp(whostring{iwho},'x','once'))>maxwhosnamel(5))
        maxwhosnamel(5) = (length(whostring{iwho})-regexp(whostring{iwho},'x','once'));
    end
end
% Pad front and back of Size title shifted to the right place.
whossizestring = sprintf('%*s%s%*s',maxwhosnamel(4)-1,'','Size',maxwhosnamel(5)+1,'');

% Adjust the Byte length for the commas.
maxwhosnamel(3) = maxwhosnamel(3)+floor(maxwhosnamel(3)/3);

maxwhosnamel(1) = max(8,maxwhosnamel(1));
maxwhosnamel(3) = max(10,maxwhosnamel(3));

% Print titles
fprintf('  %-*s     %-s     %*s   %-10s \n\n',maxwhosnamel(1),'Name',whossizestring,...
    maxwhosnamel(3)-2,'Bytes','Class');

% maxwhosnamel

% Print each line
for iwho = 1:length(Awhos)
    % Align sizes by the first x.
    whostring{iwho} = sprintf('%*s%s%*s',...
        maxwhosnamel(4)-regexp(whostring{iwho},'x','once')+1,'',whostring{iwho},...
        maxwhosnamel(5)-(length(whostring{iwho})-regexp(whostring{iwho},'x','once'))-1,'');
    
    % Add in commas to the Bytes field
    newwhosbyte = num2str(whobytes{iwho});
    for iwhol=length(newwhosbyte)-2:-3:2 % Shoul
        newwhosbyte(iwhol+1:end+1) = newwhosbyte(iwhol:end);
        newwhosbyte(iwhol) = ',';
    end
    fprintf('  %-*s     %-*s     %*s   %-10s\n',...
        maxwhosnamel(1),Awhos(iwho).name,...
        maxwhosnamel(5)+maxwhosnamel(4)+2,whostring{iwho},...
        maxwhosnamel(3),newwhosbyte,...
        Awhos(iwho).class);
end
fprintf('\n');
clear Awhos whosmaxes iwho whosize whobytes whostring iwhol whossizestring tempwhostring
clear maxwhosnamel newwhosbyte